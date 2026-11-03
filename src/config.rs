use std::env;

#[derive(Clone)]
pub struct RedisConfig {
    pub port: u16,
    pub auth: Option<String>,
    pub user: Option<String>,
    pub cluster: bool,
}

impl RedisConfig {
    pub fn from_env() -> Self {
        Self {
            port: env::var("REDIS_PORT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(6379),
            auth: env::var("REDIS_AUTH").ok(),
            user: env::var("REDIS_USER").ok(),
            cluster: env::var("REDIS_CLUSTER")
                .ok()
                .and_then(|v| v.parse::<i32>().ok())
                .map(|v| v != 0)
                .unwrap_or(false),
        }
    }

    pub fn connection_url(&self, host: &str) -> String {
        let auth_part = match (&self.user, &self.auth) {
            (Some(user), Some(pass)) => format!("{}:{}@", user, pass),
            (None, Some(pass)) => format!(":{}@", pass),
            _ => String::new(),
        };
        format!("redis://{}{}:{}/", auth_part, host, self.port)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// `from_env` reads process-global env vars, so the env-mutating tests must
    /// run serially. Pure `connection_url` tests need no guard.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn cfg(port: u16, user: Option<&str>, auth: Option<&str>, cluster: bool) -> RedisConfig {
        RedisConfig {
            port,
            auth: auth.map(String::from),
            user: user.map(String::from),
            cluster,
        }
    }

    #[test]
    fn connection_url_user_and_pass() {
        // user + pass → redis://u:p@h:port/  (order and single ':' are load-bearing
        // for ACL auth; a swap or dropped ':' silently fails against ACL Redis).
        let c = cfg(6380, Some("alice"), Some("s3cret"), false);
        assert_eq!(
            c.connection_url("myhost"),
            "redis://alice:s3cret@myhost:6380/"
        );
    }

    #[test]
    fn connection_url_pass_only() {
        // pass only → redis://:p@h:port/  (leading ':' with empty user).
        let c = cfg(6379, None, Some("pw"), false);
        assert_eq!(c.connection_url("h"), "redis://:pw@h:6379/");
    }

    #[test]
    fn connection_url_no_auth() {
        // no user, no pass → no auth segment at all.
        let c = cfg(6379, None, None, false);
        assert_eq!(c.connection_url("h"), "redis://h:6379/");
    }

    #[test]
    fn connection_url_user_without_pass_has_no_auth() {
        // user set but no pass → the `(Some, None)` case is NOT special-cased and
        // falls through to the `_ => ""` arm: no auth segment is emitted.
        let c = cfg(6379, Some("alice"), None, false);
        assert_eq!(c.connection_url("h"), "redis://h:6379/");
    }

    /// Set or clear an env var to match `val`.
    fn set_or_clear(key: &str, val: Option<&str>) {
        match val {
            Some(v) => env::set_var(key, v),
            None => env::remove_var(key),
        }
    }

    /// Run `f` with the four Redis env vars set to the given values, restoring
    /// the previous values afterwards. Serialized via `ENV_LOCK`.
    fn with_env<F: FnOnce()>(
        port: Option<&str>,
        auth: Option<&str>,
        user: Option<&str>,
        cluster: Option<&str>,
        f: F,
    ) {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved: Vec<(&str, Option<String>)> =
            ["REDIS_PORT", "REDIS_AUTH", "REDIS_USER", "REDIS_CLUSTER"]
                .iter()
                .map(|k| (*k, env::var(k).ok()))
                .collect();
        set_or_clear("REDIS_PORT", port);
        set_or_clear("REDIS_AUTH", auth);
        set_or_clear("REDIS_USER", user);
        set_or_clear("REDIS_CLUSTER", cluster);
        f();
        for (k, v) in saved {
            set_or_clear(k, v.as_deref());
        }
    }

    #[test]
    fn from_env_defaults_when_unset() {
        with_env(None, None, None, None, || {
            let c = RedisConfig::from_env();
            assert_eq!(c.port, 6379);
            assert_eq!(c.auth, None);
            assert_eq!(c.user, None);
            assert!(!c.cluster);
            assert_eq!(c.connection_url("h"), "redis://h:6379/");
        });
    }

    #[test]
    fn from_env_parses_all_fields() {
        with_env(Some("7000"), Some("pw"), Some("alice"), Some("1"), || {
            let c = RedisConfig::from_env();
            assert_eq!(c.port, 7000);
            assert_eq!(c.auth.as_deref(), Some("pw"));
            assert_eq!(c.user.as_deref(), Some("alice"));
            assert!(c.cluster);
            assert_eq!(c.connection_url("h"), "redis://alice:pw@h:7000/");
        });
    }

    #[test]
    fn from_env_cluster_coercion() {
        // REDIS_CLUSTER coerces via i32 parse then `!= 0`: "0" → false, non-zero
        // ints → true, unparseable → false (unwrap_or(false)).
        with_env(None, None, None, Some("0"), || {
            assert!(!RedisConfig::from_env().cluster);
        });
        with_env(None, None, None, Some("2"), || {
            assert!(RedisConfig::from_env().cluster);
        });
        with_env(None, None, None, Some("-1"), || {
            assert!(RedisConfig::from_env().cluster);
        });
        with_env(None, None, None, Some("true"), || {
            // "true" is not a valid i32 → parse fails → unwrap_or(false).
            assert!(!RedisConfig::from_env().cluster);
        });
    }

    #[test]
    fn from_env_bad_port_falls_back_to_default() {
        with_env(Some("not-a-port"), None, None, None, || {
            assert_eq!(RedisConfig::from_env().port, 6379);
        });
    }
}
