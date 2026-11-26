# Remove speech models and tests

## Scope

- Remove the current STT/TTS adapter, speech benchmark, and speech-specific tests.
- Remove Whisper, Edge TTS, and Piper dependencies/configuration references.
- Remove speech callbacks and controls from the Gradio app so the app remains importable.
- Update README and roadmap to describe the voice feature as removed pending a replacement design.

## Validation

- Search the repository for speech/STT/TTS/model references.
- Run the remaining test suite.
- Compile/import the app without speech dependencies.
