use crate::redis_transport::{Client, Connection};
use redis::RedisResult;

use crate::config::RedisConfig;

pub fn create_connection(host: &str, config: &RedisConfig) -> RedisResult<Connection> {
    let url = config.connection_url(host);
    let client = Client::open(url.as_str(), config.cluster)?;
    client.get_connection()
}
