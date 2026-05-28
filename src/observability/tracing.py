from __future__ import annotations


def setup_tracing() -> bool:
    """Instrument the Anthropic client with OpenInference and launch Phoenix UI.

    Requires: pip install 'rag-pipeline[tracing]'
    Phoenix dashboard: http://localhost:6006
    """
    try:
        import phoenix as px
        from openinference.instrumentation.anthropic import AnthropicInstrumentor

        session = px.launch_app()
        AnthropicInstrumentor().instrument()
        print(f"Phoenix tracing active: {session.url}")
        return True
    except ImportError:
        print("Tracing skipped (install: pip install 'rag-pipeline[tracing]')")
        return False
