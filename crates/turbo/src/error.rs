/// Generic error type for the turbo library.
/// This replaces panic!() calls in production code with proper error handling.
#[derive(Debug)]
pub struct RerfError {
    message: String,
    kind: RerfErrorKind,
}

#[derive(Debug)]
pub enum RerfErrorKind {
    ChannelClosed,
    MetadataInitFailed,
    TraceDecodingFailed,
    AddressNotCovered,
    PathProcessingFailed,
    ServerBindFailed,
    InvalidQuantile,
}

impl RerfError {
    pub fn channel_closed(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
            kind: RerfErrorKind::ChannelClosed,
        }
    }

    pub fn metadata_init_failed(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
            kind: RerfErrorKind::MetadataInitFailed,
        }
    }

    pub fn trace_decoding_failed(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
            kind: RerfErrorKind::TraceDecodingFailed,
        }
    }

    pub fn address_not_covered(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
            kind: RerfErrorKind::AddressNotCovered,
        }
    }

    pub fn path_processing_failed(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
            kind: RerfErrorKind::PathProcessingFailed,
        }
    }

    pub fn server_bind_failed(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
            kind: RerfErrorKind::ServerBindFailed,
        }
    }

    pub fn invalid_quantile(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
            kind: RerfErrorKind::InvalidQuantile,
        }
    }
}

impl std::fmt::Display for RerfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}

impl std::fmt::Display for RerfErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RerfErrorKind::ChannelClosed => write!(f, "Channel closed"),
            RerfErrorKind::MetadataInitFailed => write!(f, "Metadata initialization failed"),
            RerfErrorKind::TraceDecodingFailed => write!(f, "Trace decoding failed"),
            RerfErrorKind::AddressNotCovered => write!(f, "Address not covered"),
            RerfErrorKind::PathProcessingFailed => write!(f, "Path processing failed"),
            RerfErrorKind::ServerBindFailed => write!(f, "Server bind failed"),
            RerfErrorKind::InvalidQuantile => write!(f, "Invalid quantile"),
        }
    }
}

impl std::error::Error for RerfError {}
