# 🧪 Karpathy · Neural Networks: Zero to Hero

<p class="hero-subtitle">
Andrej Karpathy's <a href="https://karpathy.ai/zero-to-hero.html">Zero to Hero</a> series builds neural networks from scratch, starting from a single neuron and ending with GPT. It's the best resource I know for truly understanding what's happening inside these models. The concepts (tokenization, embeddings, self attention, autoregressive generation) are exactly the same ones that power audio models like MusicGen, AudioLM, and SoundStream. <strong>My approach:</strong> where I can, I swap text for audio. Where I can't, I write notes on how each concept connects to the audio world.
</p>

---

## Videos & Audio Connections

| # | Video | Core concept | Audio connection | Status |
|---|---|---|---|---|
| 1 | [Micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) | Backprop, autograd engine | Pure math. No direct audio application, but understanding gradients is essential for everything that follows. | 🔲 |
| 2 | [Makemore 1: Bigrams](https://www.youtube.com/watch?v=PaCmpygFfXo) | Character level language model | Build a bigram model that predicts the next frame of a mel spectrogram instead of the next character. First taste of autoregressive audio. | 🔲 |
| 3 | [Makemore 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I) | Multi layer perceptrons, embeddings | Embeddings are how audio tokens (EnCodec, SoundStream) get represented as vectors. Same idea as character embeddings, different domain. | 🔲 |
| 4 | [Makemore 3: BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc) | Activations, BatchNorm, diagnostics | Training dynamics. Applies the same way to audio models. Good practice with training diagnostics. | 🔲 |
| 5 | [Makemore 4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI) | Manual backprop | Pure math again. No audio twist needed, just do it and understand it. | 🔲 |
| 6 | [Makemore 5: WaveNet](https://www.youtube.com/watch?v=t3YJ5hKiMQ0) | Dilated causal convolutions | WaveNet was originally built for audio! This is where Karpathy's series and audio generation directly intersect. Implement it on actual waveforms. | 🔲 |
| 7 | [GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) | Self attention, transformer, autoregressive generation | This is the architecture behind MusicGen and AudioLM. Build GPT on text, then think about what changes when your tokens are audio codes instead of words. | 🔲 |
| 8 | [Tokenization](https://www.youtube.com/watch?v=zduSFxRajkE) | BPE, tokenizers | Directly connects to EnCodec/SoundStream: how do you turn continuous audio into discrete tokens? BPE on text ↔ neural codec on audio. Same problem, different modality. | 🔲 |
| 9 | [Reproducing GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU) | Full training pipeline at scale | The full picture. Training infrastructure, data loading, optimization. Everything you'd need to train an audio GPT. | 🔲 |

---

### Which videos translate best to audio?

**Direct audio application** (build something that makes sound): Makemore 1 (bigram on audio frames), Makemore 5/WaveNet (literally an audio model), GPT from scratch (the MusicGen architecture).

**Conceptual bridge** (understand the idea, connect it to audio): Tokenization (BPE ↔ neural audio codecs), Makemore 2 (embeddings ↔ audio token embeddings).

**Pure foundations** (just learn it, it applies everywhere): Micrograd, Makemore 3 and 4, Reproducing GPT-2.

---

_This page will grow as I work through the series._
