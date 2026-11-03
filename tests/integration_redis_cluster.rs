//! Only tests/run-redis-cluster.sh supplies the ephemeral password; all nodes are
//! loopback fixtures. Never run cluster-admin tests against a managed endpoint.
use redis::{ConnectionLike, Value};
use vector_db_benchmark::redis_transport::Client;

fn fixture() -> (Client, Vec<redis::Connection>) {
    let port: u16 = std::env::var("REDIS_CLUSTER_TEST_PORT")
        .expect("use make integration-test-redis-cluster")
        .parse()
        .unwrap();
    assert_eq!(port, 17379);
    let password =
        std::env::var("REDIS_CLUSTER_TEST_PASSWORD").expect("owned fixture password required");
    assert!(password.starts_with("fixture-") && password.len() == 32);
    let url = |p| format!("redis://:{password}@127.0.0.1:{p}/");
    let nodes = (0..3)
        .map(|i| {
            redis::Client::open(url(port + i))
                .unwrap()
                .get_connection()
                .unwrap()
        })
        .collect();
    (Client::open(&url(port), true).unwrap(), nodes)
}
fn slot(conn: &mut redis::Connection, key: &str) -> u16 {
    redis::cmd("CLUSTER")
        .arg("KEYSLOT")
        .arg(key)
        .query(conn)
        .unwrap()
}
fn keys_for_nodes(conn: &mut redis::Connection, prefix: &str) -> Vec<String> {
    let mut keys = vec![String::new(); 3];
    for i in 0..10000 {
        let key = format!("{prefix}{i}");
        let s = slot(conn, &key);
        let node = if s < 5461 {
            0
        } else if s < 10923 {
            1
        } else {
            2
        };
        if keys[node].is_empty() {
            keys[node] = key;
        }
        if keys.iter().all(|s| !s.is_empty()) {
            return keys;
        }
    }
    panic!("unable to spread fixture slots")
}

#[test]
fn cluster_pipeline_routes_every_primary_and_preserves_payloads_and_errors() {
    let (client, mut nodes) = fixture();
    let mut conn = client.get_connection().unwrap();
    let keys = keys_for_nodes(&mut nodes[0], "pipeline:");
    let commands = keys
        .iter()
        .enumerate()
        .map(|(i, key)| {
            let mut cmd = redis::cmd("HSET");
            cmd.arg(key)
                .arg("vector")
                .arg(&[0u8, 255, i as u8])
                .arg("doc_len")
                .arg(i);
            cmd
        })
        .collect::<Vec<_>>();
    let replies = conn.pipeline(&commands).unwrap();
    assert_eq!(replies, vec![Value::Int(2); 3]);
    for (i, key) in keys.iter().enumerate() {
        let value: Vec<u8> = redis::cmd("HGET")
            .arg(key)
            .arg("vector")
            .query(&mut conn)
            .unwrap();
        assert_eq!(value, vec![0, 255, i as u8]);
        assert_eq!(
            redis::cmd("HGET")
                .arg(key)
                .arg("doc_len")
                .query::<usize>(&mut conn)
                .unwrap(),
            i
        );
    }
    // Response order must follow commands, not node-group completion order.
    let gets = keys
        .iter()
        .enumerate()
        .rev()
        .map(|(_, key)| {
            let mut c = redis::cmd("HGET");
            c.arg(key).arg("doc_len");
            c
        })
        .collect::<Vec<_>>();
    let values: Vec<usize> = conn
        .pipeline(&gets)
        .unwrap()
        .into_iter()
        .map(|v| redis::from_redis_value(v).unwrap())
        .collect();
    assert_eq!(values, vec![2, 1, 0]);
    redis::cmd("SET")
        .arg(&keys[1])
        .arg("wrongtype")
        .query::<()>(&mut conn)
        .unwrap();
    assert!(conn.pipeline(&commands).is_err());
    // Standard Pipeline must fail instead of sending all slots to one primary.
    assert!(redis::pipe()
        .cmd("SET")
        .arg("ordinary:0")
        .arg(1)
        .query::<()>(&mut conn)
        .is_err());
    for key in keys {
        redis::cmd("DEL").arg(key).query::<()>(&mut conn).unwrap();
    }
}

#[test]
fn prefix_scan_and_markers_cover_nonseed_primaries_with_resp3() {
    let (client, mut nodes) = fixture();
    let mut conn = client
        .with_protocol(redis::ProtocolVersion::RESP3)
        .get_connection()
        .unwrap();
    let keys = keys_for_nodes(&mut nodes[0], "owned[*]:");
    let marker = keys_for_nodes(&mut nodes[0], "vdbb:fulltext-profile:fixture:")[2].clone();
    redis::cmd("SET")
        .arg(&marker)
        .arg("profile-v1")
        .query::<()>(&mut conn)
        .unwrap();
    assert_eq!(
        redis::cmd("GET")
            .arg(&marker)
            .query::<String>(&mut conn)
            .unwrap(),
        "profile-v1"
    );
    assert_eq!(conn.scan_prefix("owned[*]:", false).unwrap(), 0);
    redis::cmd("SET")
        .arg(&keys[2])
        .arg("nonseed")
        .query::<()>(&mut conn)
        .unwrap();
    assert_eq!(conn.scan_prefix("owned[*]:", true).unwrap(), 1);
    for key in &keys {
        redis::cmd("SET")
            .arg(key)
            .arg("payload")
            .query::<()>(&mut conn)
            .unwrap();
    }
    assert_eq!(conn.scan_prefix("owned[*]:", false).unwrap(), 3);
    for spelling in ["SCAN", "scan", "ScAn"] {
        assert!(redis::cmd(spelling)
            .arg(0)
            .query::<Value>(&mut conn)
            .is_err());
    }
    redis::cmd("ACL")
        .arg("SETUSER")
        .arg("default")
        .arg("-scan")
        .query::<()>(&mut nodes[2])
        .unwrap();
    let denied = conn.scan_prefix("owned[*]:", false);
    redis::cmd("ACL")
        .arg("SETUSER")
        .arg("default")
        .arg("+scan")
        .query::<()>(&mut nodes[2])
        .unwrap();
    assert!(
        denied.is_err(),
        "a partial scan must never certify a corpus"
    );
    for key in keys.iter().chain(std::iter::once(&marker)) {
        redis::cmd("DEL").arg(key).query::<()>(&mut conn).unwrap();
    }
    assert!(conn.check_connection());
}

#[test]
fn moved_and_ask_are_followed_by_an_existing_connection() {
    let (client, mut nodes) = fixture();
    let mut conn = client.get_connection().unwrap();
    let key = keys_for_nodes(&mut nodes[0], "redirect:")[0].clone();
    let hash_slot = slot(&mut nodes[0], &key);
    let ids = nodes
        .iter_mut()
        .map(|n| {
            redis::cmd("CLUSTER")
                .arg("MYID")
                .query::<String>(n)
                .unwrap()
        })
        .collect::<Vec<_>>();
    redis::cmd("SET")
        .arg(&key)
        .arg("before")
        .query::<()>(&mut conn)
        .unwrap();
    redis::cmd("CLUSTER")
        .arg("SETSLOT")
        .arg(hash_slot)
        .arg("IMPORTING")
        .arg(&ids[0])
        .query::<()>(&mut nodes[1])
        .unwrap();
    redis::cmd("CLUSTER")
        .arg("SETSLOT")
        .arg(hash_slot)
        .arg("MIGRATING")
        .arg(&ids[1])
        .query::<()>(&mut nodes[0])
        .unwrap();
    // Move just our owned key; do not change other slots or wipe any database.
    let password = std::env::var("REDIS_CLUSTER_TEST_PASSWORD").unwrap();
    redis::cmd("MIGRATE")
        .arg("127.0.0.1")
        .arg(17380)
        .arg(&key)
        .arg(0)
        .arg(5000)
        .arg("AUTH")
        .arg(password)
        .query::<()>(&mut nodes[0])
        .unwrap();
    let ask = redis::cmd("GET")
        .arg(&key)
        .query::<Value>(&mut nodes[0])
        .unwrap_err();
    assert_eq!(ask.code(), Some("ASK"));
    assert_eq!(
        redis::cmd("GET")
            .arg(&key)
            .query::<String>(&mut conn)
            .unwrap(),
        "before"
    );
    redis::cmd("SET")
        .arg(&key)
        .arg("after-ask")
        .query::<()>(&mut conn)
        .unwrap();
    for node in &mut nodes {
        redis::cmd("CLUSTER")
            .arg("SETSLOT")
            .arg(hash_slot)
            .arg("NODE")
            .arg(&ids[1])
            .query::<()>(node)
            .unwrap();
    }
    let moved = redis::cmd("GET")
        .arg(&key)
        .query::<Value>(&mut nodes[0])
        .unwrap_err();
    assert_eq!(moved.code(), Some("MOVED"));
    assert_eq!(
        redis::cmd("GET")
            .arg(&key)
            .query::<String>(&mut conn)
            .unwrap(),
        "after-ask"
    );
    redis::cmd("DEL").arg(&key).query::<()>(&mut conn).unwrap();
    for node in &mut nodes {
        redis::cmd("CLUSTER")
            .arg("SETSLOT")
            .arg(hash_slot)
            .arg("NODE")
            .arg(&ids[0])
            .query::<()>(node)
            .unwrap();
    }
}

#[test]
fn migration_between_nonseed_nodes_refuses_even_an_empty_prefix() {
    let (client, mut nodes) = fixture();
    let mut conn = client.get_connection().unwrap();
    let key = keys_for_nodes(&mut nodes[0], "migration-nonseed:")[1].clone();
    let s = slot(&mut nodes[0], &key);
    let ids = nodes
        .iter_mut()
        .map(|n| {
            redis::cmd("CLUSTER")
                .arg("MYID")
                .query::<String>(n)
                .unwrap()
        })
        .collect::<Vec<_>>();
    redis::cmd("CLUSTER")
        .arg("SETSLOT")
        .arg(s)
        .arg("IMPORTING")
        .arg(&ids[1])
        .query::<()>(&mut nodes[2])
        .unwrap();
    redis::cmd("CLUSTER")
        .arg("SETSLOT")
        .arg(s)
        .arg("MIGRATING")
        .arg(&ids[2])
        .query::<()>(&mut nodes[1])
        .unwrap();
    let result = conn.scan_prefix("no-such-prefix:", false);
    for i in [1, 2] {
        redis::cmd("CLUSTER")
            .arg("SETSLOT")
            .arg(s)
            .arg("STABLE")
            .query::<()>(&mut nodes[i])
            .unwrap();
    }
    assert!(
        result.is_err(),
        "nonseed migration must not certify an empty prefix"
    );
}
