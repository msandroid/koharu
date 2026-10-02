use keyring_core::Entry;
use secrecy::{SecretBox, zeroize::Zeroize};
use serde::{Deserialize, Serialize, Serializer};
use std::sync::LazyLock;

pub use secrecy::{ExposeSecret, SerializableSecret};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(transparent)]
pub struct SecretString(SecretBox<SecretValue>);

impl Default for SecretString {
    fn default() -> Self {
        Self::from(String::new())
    }
}

impl From<String> for SecretString {
    fn from(value: String) -> Self {
        Self(SecretBox::new(Box::new(SecretValue(value))))
    }
}

impl From<&str> for SecretString {
    fn from(value: &str) -> Self {
        Self::from(value.to_owned())
    }
}

impl ExposeSecret<str> for SecretString {
    fn expose_secret(&self) -> &str {
        &self.0.expose_secret().0
    }
}

#[derive(Clone, Deserialize)]
#[serde(transparent)]
struct SecretValue(String);

impl Zeroize for SecretValue {
    fn zeroize(&mut self) {
        self.0.zeroize();
    }
}

impl secrecy::CloneableSecret for SecretValue {}

impl Serialize for SecretValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("[REDACTED]")
    }
}

impl SerializableSecret for SecretValue {}

const SERVICE: &str = "koharu";

/// Platforms whose credential store is registered with `keyring_core` here
/// rather than chosen by `keyring`'s defaults.
const EXPLICIT_STORE: bool = cfg!(any(
    target_os = "linux",
    target_os = "android",
    target_os = "ios"
));

static CREDENTIAL_STORE: LazyLock<Result<(), String>> = LazyLock::new(|| {
    #[cfg(target_os = "linux")]
    let store = linux_keyutils_keyring_store::Store::new();
    // The data protection keychain is the only keychain on iOS.
    #[cfg(target_os = "ios")]
    let store = apple_native_keyring_store::protected::Store::new();
    // Requires the `ndk-context` application context, which Tauri's Android
    // runtime initializes before app code runs.
    #[cfg(target_os = "android")]
    let store = android_native_keyring_store::Store::new();
    #[cfg(any(target_os = "linux", target_os = "android", target_os = "ios"))]
    return store
        .map(|store| keyring_core::set_default_store(store))
        .map_err(|error| error.to_string());
    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "ios")))]
    Ok(())
});

/// Load a Koharu secret by key, returning `None` when no credential exists.
pub fn get(key: &str) -> anyhow::Result<Option<SecretString>> {
    let entry = entry(key)?;
    match entry.get_password() {
        Ok(value) => Ok(Some(SecretString::from(value))),
        Err(keyring_core::Error::NoEntry) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

/// Store a Koharu secret by key.
pub fn set(key: &str, secret: &SecretString) -> anyhow::Result<()> {
    entry(key)?.set_password(secret.expose_secret())?;
    Ok(())
}

/// Delete a Koharu secret by key. Missing credentials are treated as success.
pub fn delete(key: &str) -> anyhow::Result<()> {
    match entry(key)?.delete_credential() {
        Ok(()) | Err(keyring_core::Error::NoEntry) => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn entry(key: &str) -> anyhow::Result<Entry> {
    match &*CREDENTIAL_STORE {
        Ok(()) => {}
        Err(error) => anyhow::bail!("failed to initialize the credential store: {error}"),
    }
    if EXPLICIT_STORE {
        Ok(Entry::new(SERVICE, key)?)
    } else {
        Ok(keyring::Entry::new(SERVICE, key)?.inner)
    }
}
