//! Redis engine transport: keyed commands use primary slot routing; Search
//! commands use the database coordinator exactly once. No replica reads.
use redis::{
    cluster::{ClusterClient, ClusterConnection},
    Cmd, ConnectionInfo, ConnectionLike, RedisResult, Value,
};
use std::collections::{BTreeSet, HashSet};
use std::time::{Duration, Instant};

fn error(message: &'static str) -> redis::RedisError {
    (redis::ErrorKind::Client, message).into()
}

pub fn resolve_cluster_api(
    value: Option<&serde_json::Value>,
    environment: Option<&str>,
) -> Result<bool, String> {
    if let Some(value) = value {
        return value
            .as_bool()
            .ok_or_else(|| "collection_params.cluster_api must be boolean".into());
    }
    match environment {
        None | Some("0" | "false") => Ok(false),
        Some("1" | "true") => Ok(true),
        Some(_) => Err("REDIS_CLUSTER must be 0/1 or false/true".into()),
    }
}

#[derive(Clone)]
pub struct Client {
    info: ConnectionInfo,
    cluster_api: bool,
}
impl Client {
    pub fn open(url: &str, cluster_api: bool) -> RedisResult<Self> {
        let info = redis::Client::open(url)?.get_connection_info().clone();
        if cluster_api && info.redis_settings().db() != 0 {
            return Err(error("OSS Cluster API requires database 0"));
        }
        Ok(Self { info, cluster_api })
    }
    pub fn with_protocol(mut self, protocol: redis::ProtocolVersion) -> Self {
        let settings = self.info.redis_settings().clone().set_protocol(protocol);
        self.info = self.info.set_redis_settings(settings);
        self
    }
    pub fn get_connection(&self) -> RedisResult<Connection> {
        let coordinator = redis::Client::open(self.info.clone())?
            .get_connection_with_timeout(Duration::from_secs(5))?;
        coordinator.set_read_timeout(Some(Duration::from_secs(120)))?;
        coordinator.set_write_timeout(Some(Duration::from_secs(120)))?;
        let cluster = if self.cluster_api {
            Some(
                ClusterClient::builder([self.info.clone()])
                    .retries(3)
                    .min_retry_wait(50)
                    .max_retry_wait(500)
                    .connection_timeout(Duration::from_secs(5))
                    .response_timeout(Duration::from_secs(30))
                    .build()?
                    .get_connection()?,
            )
        } else {
            None
        };
        Ok(Connection {
            coordinator,
            cluster,
        })
    }
}

pub struct Connection {
    coordinator: redis::Connection,
    cluster: Option<ClusterConnection>,
}
impl Connection {
    pub fn set_read_timeout(&self, duration: Option<Duration>) -> RedisResult<()> {
        self.coordinator.set_read_timeout(duration)?;
        if let Some(cluster) = &self.cluster {
            cluster.set_read_timeout(duration)?;
        }
        Ok(())
    }
    pub fn set_write_timeout(&self, duration: Option<Duration>) -> RedisResult<()> {
        self.coordinator.set_write_timeout(duration)?;
        if let Some(cluster) = &self.cluster {
            cluster.set_write_timeout(duration)?;
        }
        Ok(())
    }
    /// Ordinary redis::pipe is not multi-slot safe. Use the upstream
    /// dedicated cluster pipeline, which partitions commands and restores order.
    pub fn pipeline(&mut self, commands: &[Cmd]) -> RedisResult<Vec<Value>> {
        if let Some(cluster) = &mut self.cluster {
            if commands.iter().any(|c| {
                coordinator_command(c)
                    || command_name(c).is_some_and(|n| n.eq_ignore_ascii_case(b"SCAN"))
            }) {
                return Err(error(
                    "Cluster pipelines are for keyed commands, not coordinator commands or SCAN",
                ));
            }
            let mut pipe = redis::cluster::cluster_pipe();
            for command in commands {
                pipe.add_command(command.clone());
            }
            pipe.query(cluster)
        } else {
            let mut pipe = redis::pipe();
            for command in commands {
                pipe.add_command(command.clone());
            }
            pipe.query(&mut self.coordinator)
        }
    }
    fn topology(&mut self) -> RedisResult<Vec<SlotRange>> {
        let value: Value = redis::cmd("CLUSTER")
            .arg("SLOTS")
            .query(&mut self.coordinator)?;
        let ranges = parse_topology(&value)?;
        let primaries: BTreeSet<_> = ranges.iter().map(|r| (r.host.clone(), r.port)).collect();
        // Migration annotations belong to each node's `myself` line. A seed's
        // view alone cannot detect migration between two other primaries.
        for (host, port) in primaries {
            let mut command = redis::cmd("CLUSTER");
            command.arg("NODES");
            let value = self.cluster.as_mut().unwrap().route_command(
                &command,
                redis::cluster_routing::RoutingInfo::SingleNode(
                    redis::cluster_routing::SingleNodeRoutingInfo::ByAddress { host, port },
                ),
            )?;
            let nodes: String = redis::from_redis_value(value)?;
            if nodes.contains("->-") || nodes.contains("-<-") {
                return Err(error(
                    "Cluster slot migration prevents a stable prefix scan",
                ));
            }
        }
        // A migration may finish while NODES is being read: its annotations
        // disappear, but the slot map must still match our initial snapshot.
        let after: Value = redis::cmd("CLUSTER")
            .arg("SLOTS")
            .query(&mut self.coordinator)?;
        ensure_same_topology(&ranges, &parse_topology(&after)?)?;
        Ok(ranges)
    }
    /// Fail closed on partial scans, duplicate pages or changed slot ownership.
    /// Fresh-prefix checks stop at the first key; complete counts deduplicate.
    pub fn scan_prefix(&mut self, prefix: &str, stop_on_first: bool) -> RedisResult<usize> {
        let pattern = escape_glob(prefix) + "*";
        let topology = if self.cluster.is_some() {
            Some(self.topology()?)
        } else {
            None
        };
        let nodes: BTreeSet<_> = topology
            .as_ref()
            .into_iter()
            .flatten()
            .map(|s| (s.host.clone(), s.port))
            .collect();
        let targets: Vec<_> = if topology.is_some() {
            nodes.into_iter().map(Some).collect()
        } else {
            vec![None]
        };
        let start = Instant::now();
        let mut keys = HashSet::<Vec<u8>>::new();
        for target in targets {
            let mut cursor = 0u64;
            let mut seen = HashSet::new();
            loop {
                if start.elapsed() > Duration::from_secs(300)
                    || !seen.insert(cursor)
                    || seen.len() > 1_000_000
                {
                    return Err(error(
                        "Prefix scan exceeded its time/page budget or repeated a cursor",
                    ));
                }
                let mut cmd = redis::cmd("SCAN");
                cmd.arg(cursor)
                    .arg("MATCH")
                    .arg(&pattern)
                    .arg("COUNT")
                    .arg(1000);
                let value = if let Some((host, port)) = &target {
                    self.cluster.as_mut().unwrap().route_command(
                        &cmd,
                        redis::cluster_routing::RoutingInfo::SingleNode(
                            redis::cluster_routing::SingleNodeRoutingInfo::ByAddress {
                                host: host.clone(),
                                port: *port,
                            },
                        ),
                    )?
                } else {
                    self.coordinator.req_command(&cmd)?
                };
                let (next, page): (u64, Vec<Vec<u8>>) = redis::from_redis_value(value)?;
                if stop_on_first && !page.is_empty() {
                    return Ok(1);
                }
                keys.extend(page);
                if next == 0 {
                    break;
                }
                cursor = next;
            }
        }
        if let Some(before) = topology {
            ensure_same_topology(&before, &self.topology()?)?;
        }
        Ok(keys.len())
    }
}

fn command_name(cmd: &Cmd) -> Option<&[u8]> {
    match cmd.args_iter().next() {
        Some(redis::Arg::Simple(name)) => Some(name),
        _ => None,
    }
}
fn coordinator_command(cmd: &Cmd) -> bool {
    command_name(cmd).is_some_and(|name| {
        name.get(..3)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case(b"FT."))
            || [
                b"INFO".as_slice(),
                b"CONFIG",
                b"MODULE",
                b"PING",
                b"CLUSTER",
            ]
            .iter()
            .any(|n| name.eq_ignore_ascii_case(n))
    })
}
impl ConnectionLike for Connection {
    fn req_command(&mut self, cmd: &Cmd) -> RedisResult<Value> {
        if self.cluster.is_none() || coordinator_command(cmd) {
            return self.coordinator.req_command(cmd);
        }
        if command_name(cmd).is_some_and(|name| name.eq_ignore_ascii_case(b"SCAN")) {
            return Err(error(
                "Cluster SCAN requires scan_prefix over every primary",
            ));
        }
        self.cluster.as_mut().unwrap().req_command(cmd)
    }
    fn req_packed_command(&mut self, bytes: &[u8]) -> RedisResult<Value> {
        if self.cluster.is_none() {
            return self.coordinator.req_packed_command(bytes);
        }
        // Normal Cmd::query uses req_command. Refuse raw bytes rather than
        // silently guessing command/coordinator routing from an opaque buffer.
        Err(error(
            "Raw packed commands are unsupported in cluster mode; use Cmd::query",
        ))
    }
    fn req_packed_commands(
        &mut self,
        bytes: &[u8],
        offset: usize,
        count: usize,
    ) -> RedisResult<Vec<Value>> {
        if self.cluster.is_none() {
            return self.coordinator.req_packed_commands(bytes, offset, count);
        }
        Err(error(
            "Use the dedicated cluster pipeline for multi-slot commands",
        ))
    }
    fn supports_pipelining(&self) -> bool {
        self.cluster.is_none()
    }
    fn get_db(&self) -> i64 {
        self.coordinator.get_db()
    }
    fn is_open(&self) -> bool {
        self.coordinator.is_open() && self.cluster.as_ref().is_none_or(ConnectionLike::is_open)
    }
    fn check_connection(&mut self) -> bool {
        self.coordinator.check_connection()
            && self
                .cluster
                .as_mut()
                .is_none_or(ConnectionLike::check_connection)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SlotRange {
    start: u16,
    end: u16,
    host: String,
    port: u16,
    id: String,
}
fn ensure_same_topology(before: &[SlotRange], after: &[SlotRange]) -> RedisResult<()> {
    if before != after {
        return Err(error("Cluster topology changed during prefix scan"));
    }
    Ok(())
}
fn parse_topology(value: &Value) -> RedisResult<Vec<SlotRange>> {
    let Value::Array(rows) = value else {
        return Err(error("CLUSTER SLOTS must return ranges"));
    };
    let mut ranges = Vec::new();
    for row in rows {
        let Value::Array(row) = row else {
            return Err(error("Invalid slot range"));
        };
        if row.len() < 3 {
            return Err(error("Missing primary in slot range"));
        }
        let Value::Array(node) = &row[2] else {
            return Err(error("Invalid primary"));
        };
        if node.len() < 3 {
            return Err(error("Missing primary identity"));
        }
        let range = SlotRange {
            start: redis::from_redis_value(row[0].clone())?,
            end: redis::from_redis_value(row[1].clone())?,
            host: redis::from_redis_value(node[0].clone())?,
            port: redis::from_redis_value(node[1].clone())?,
            id: redis::from_redis_value(node[2].clone())?,
        };
        if range.host.is_empty()
            || range.id.is_empty()
            || range.port == 0
            || range.end < range.start
            || range.end >= 16384
        {
            return Err(error("Invalid cluster primary or slot bounds"));
        }
        ranges.push(range);
    }
    ranges.sort_by_key(|r| r.start);
    let mut next = 0usize;
    for r in &ranges {
        if usize::from(r.start) != next {
            return Err(error("Incomplete or overlapping cluster slot map"));
        }
        next = usize::from(r.end) + 1;
    }
    if next != 16384 {
        return Err(error("Incomplete cluster slot coverage"));
    }
    Ok(ranges)
}
fn escape_glob(prefix: &str) -> String {
    prefix
        .chars()
        .flat_map(|c| {
            if matches!(c, '*' | '?' | '[' | ']' | '\\') {
                vec!['\\', c]
            } else {
                vec![c]
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_cluster_mode_is_strict_and_overrides_legacy_environment() {
        assert!(!resolve_cluster_api(None, None).unwrap());
        assert!(resolve_cluster_api(None, Some("1")).unwrap());
        assert!(!resolve_cluster_api(Some(&serde_json::json!(false)), Some("1")).unwrap());
        assert!(resolve_cluster_api(Some(&serde_json::json!("true")), None).is_err());
        assert!(resolve_cluster_api(None, Some("yes")).is_err());
    }
    #[test]
    fn search_is_coordinated_once_but_markers_and_hashes_are_key_routed() {
        for name in [
            "FT.CREATE",
            "FT.SEARCH",
            "FT.INFO",
            "FT._LIST",
            "FT.DROPINDEX",
            "FT.PROFILE",
            "INFO",
            "CONFIG",
        ] {
            assert!(coordinator_command(&redis::cmd(name)), "{name}");
        }
        for name in ["SET", "GET", "DEL", "HSET", "HGET", "SCAN"] {
            assert!(!coordinator_command(&redis::cmd(name)), "{name}");
        }
        assert_eq!(escape_glob("a[*]?\\:"), "a\\[\\*\\]\\?\\\\:");
    }
    #[test]
    fn topology_requires_every_slot_and_does_not_count_replicas() {
        let node = Value::Array(vec![
            Value::BulkString(b"127.0.0.1".to_vec()),
            Value::Int(6379),
            Value::BulkString(b"primary".to_vec()),
        ]);
        let range = |start, end| {
            Value::Array(vec![
                Value::Int(start),
                Value::Int(end),
                node.clone(),
                node.clone(),
            ])
        };
        assert_eq!(
            parse_topology(&Value::Array(vec![range(0, 16383)]))
                .unwrap()
                .len(),
            1
        );
        for rows in [
            vec![],
            vec![range(1, 16383)],
            vec![range(0, 16000)],
            vec![range(0, 8000), range(8000, 16383)],
        ] {
            assert!(parse_topology(&Value::Array(rows)).is_err());
        }
        let before = parse_topology(&Value::Array(vec![range(0, 16383)])).unwrap();
        assert!(ensure_same_topology(&before, &before).is_ok());
        let mut after = before.clone();
        after[0].id = "replacement-primary".into();
        assert!(ensure_same_topology(&before, &after).is_err());
        after = before.clone();
        after[0].port += 1;
        assert!(ensure_same_topology(&before, &after).is_err());
    }
}
