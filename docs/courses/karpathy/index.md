# 🧪 Karpathy · Neural Networks: Zero to Hero

<p class="hero-subtitle">
Andrej Karpathy's <a href="https://karpathy.ai/zero-to-hero.html">Zero to Hero</a> series builds neural networks from scratch, starting from a single neuron and ending with GPT. It's the best resource I know for truly understanding what's happening inside these models. The concepts (tokenization, embeddings, self attention, autoregressive generation) are exactly the same ones that power audio models like MusicGen, AudioLM, and SoundStream. <strong>My approach:</strong> where I can, I swap text for audio. Where I can't, I write notes on how each concept connects to the audio world.
</p>

---

## Lectures

| # | Video | Audio connection | Status |
|---|---|---|---|
| 1 | [Micrograd](lecture1.md) | Built a soundscape classifier (rain/wind/forest) with hand crafted audio features on our micrograd MLP | ✅ Done |
| 2 | [Makemore 1: Bigrams](https://www.youtube.com/watch?v=PaCmpygFfXo) | Bigram model on audio frames instead of characters. First taste of autoregressive audio. | 🔲 |
| 3 | [Makemore 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I) | Embeddings are how audio tokens (EnCodec, SoundStream) get represented as vectors. | 🔲 |
| 4 | [Makemore 3: BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc) | Training dynamics. Same principles apply to audio models. | 🔲 |
| 5 | [Makemore 4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI) | Pure math. Just do it and understand it. | 🔲 |
| 6 | [Makemore 5: WaveNet](https://www.youtube.com/watch?v=t3YJ5hKiMQ0) | WaveNet was originally built for audio! Implement dilated causal convolutions on actual waveforms. | 🔲 |
| 7 | [GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) | The architecture behind MusicGen and AudioLM. Build GPT on text, then think about audio tokens. | 🔲 |
| 8 | [Tokenization](https://www.youtube.com/watch?v=zduSFxRajkE) | BPE on text ↔ neural codec on audio (EnCodec/SoundStream). Same problem, different modality. | 🔲 |
| 9 | [Reproducing GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU) | Full training pipeline at scale. Everything you'd need to train an audio GPT. | 🔲 |

---

### Which lectures translate best to audio?

**Direct audio application** (build something that makes sound): Lecture 1 (done!), Makemore 1 (bigram on audio frames), Makemore 5/WaveNet (literally an audio model), GPT from scratch (the MusicGen architecture).

**Conceptual bridge** (understand the idea, connect it to audio): Tokenization (BPE ↔ neural audio codecs), Makemore 2 (embeddings ↔ audio token embeddings).

**Pure foundations** (just learn it, it applies everywhere): Micrograd, Makemore 3 and 4, Reproducing GPT-2.

---

_This page will grow as I work through the series._
