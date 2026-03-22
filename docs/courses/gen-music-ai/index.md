# 🎵 Generative Music AI — Course Notes

<p class="hero-subtitle">
Notes from Valerio Velardo's <a href="https://github.com/musikalkemist/generativemusicaicourse">Generative Music AI Course</a> (The Sound of AI, 2023). 22 lectures covering the full spectrum: from rule based symbolic generation to neural audio synthesis. Produced in collaboration with the Music Technology Group at Universitat Pompeu Fabra (UPF) in Barcelona. Three instructors: Valerio Velardo, Iran Roman (NYU), and Xavier Serra (MTG UPF). My notes exist so I can look up a concept in 30 seconds instead of rewatching a video.
</p>

📺 **[Video series](https://www.youtube.com/@ValerioVelwordo/playlists)** · 💻 **[GitHub repo](https://github.com/musikalkemist/generativemusicaicourse)** · Dependencies: `music21==9.9.1`, `tensorflow==2.20.0`

---

## Part 1: Foundations (Lectures 1 to 8)

### What is generative music?

A generative music system is software that automates, to varying degrees, the composition and production of music. The key word is *automates*. Most useful systems live on a spectrum between full human control and full machine autonomy. Brian Eno coined the term in 1996. His concept with Music for Airports (1978) was to build a system that never plays the same thing twice. It was a philosophical position before it was a technical one.

Four dimensions to characterize any system: **autonomy** (how much creative control does the system have), **interactivity** (can you steer it in real time), **adaptivity** (does it respond to external signals like game state or emotion), and **originality** (recombination vs genuinely new patterns).

### History: five eras

Every time a new compute infrastructure arrives, generative music makes a step change. Not linear, it jumps.

| Era | Period | Key milestone |
|---|---|---|
| Pre computer | 1700 to 1956 | Mozart's Musikalisches Würfelspiel (1787): generated minuets by rolling dice |
| Academic | 1957 to 2009 | ILLIAC Suite (first computer generated score), Xenakis, MAX/MSP at IRCAM |
| First startup wave | 2010 to 2016 | Jukedeck, Amper Music, Melodrive. Rule based plus early ML, commercial focus |
| Big tech experiments | 2017 to 2022 | Google Magenta, MuseNet (GPT 2 on music), Jukebox (VQ VAE raw audio) |
| Music AI hype | 2023+ | MusicLM, MusicGen, Suno, Udio. Text to music for everyone. First copyright lawsuits |

### Use cases

Four domains with real commercial traction: **interactive media** (games, VR, music that adapts to game state), **content creation** (YouTube, podcasts, royalty free at scale), **composer tools** (AI suggests chords, completes melodies, harmonizes lines), and **therapeutic/ambient** (meditation, sleep, always slightly different, never jarring).

When does generative music make sense? High volume (too much to commission), adaptivity required (music must respond to context), the musicality bar is functional not artistic, and cost sensitivity.

### Symbolic vs Audio generation

This is the most important design decision when building a system.

| Dimension | Symbolic (MIDI) | Audio (Waveform) |
|---|---|---|
| Representation | Notes, durations, chords | Raw audio samples or spectrogram |
| Data volume | Small | Very large |
| Interpretability | High, editable in a DAW | Black box output |
| Timbre / expressiveness | Limited | Complete |
| Music theory integration | Easy | Hard |
| Examples | MuseNet, Magenta, Music Transformer | RAVE, Jukebox, MusicGen |

The current trend is hybrid: generate symbolic structure with a Transformer, then synthesize with a neural audio model like RAVE.

### Generative techniques taxonomy

Rule based methods (grammars, Markov chains, cellular automata, genetic algorithms) are controllable and interpretable but require hand crafted knowledge. ML methods (RNNs, Transformers, VAEs, diffusion) learn complex patterns from data but are hard to steer. Hybrid approaches try to get both. That's where active research is happening.

### What current systems still can't do

**Long range coherence**: systems generate locally convincing music but lose global structure. No model today reliably builds a compelling 5 minute musical arc. **Music theory ignorance**: most models learn statistics, not harmonic logic. **Controllability**: saying "make it more tense here, then resolve" is still very hard. **Evaluation**: there is no good objective metric for musical quality.

---

## Part 2: Technical Dive (Lectures 9 to 19)

### Generative Grammars and L Systems (Lectures 9 to 10)

A formal grammar defines rewrite rules that expand symbols into sequences. L Systems (Lindenmayer systems, originally for modelling plant growth) produce self similar, fractal like structures when mapped to music. Each letter maps to a chord or note.

```python
class LSystem:
    def __init__(self, axiom, rules):
        self.axiom = axiom
        self.rules = rules
        self.output = axiom

    def iterate(self, n=1):
        for i in range(n):
            self.output = "".join(
                self.rules.get(s, s) for s in self.output
            )
        return self.output

# Axiom: "A", Rules: A→ABC, B→BA, C→EF, F→GFD
# After 4 iterations: a fractal chord progression
# Each letter maps to a diatonic chord in C major
```

### Markov Chains (Lectures 11 to 12)

A Markov chain models next state probability based only on the current state. Applied to melody: given the note I'm playing, what note is most likely next? The probabilities are learned from data (a corpus of MIDI melodies) instead of hand written. Each state is a `(pitch, duration)` pair, so the chain also learns rhythmic patterns.

```python
class MarkovChainMelodyGenerator:
    def __init__(self, states):
        self.states = states
        self.transition_matrix = np.zeros((len(states), len(states)))

    def train(self, notes):
        # Build transition matrix from training data
        # Each row sums to 1

    def generate(self, length):
        melody = [self._generate_starting_state()]
        for _ in range(1, length):
            melody.append(self._generate_next_state(melody[-1]))
        return melody

# Training data: "Twinkle Twinkle Little Star"
# States: [("C5", 1), ("D5", 1), ("E5", 1), ("G5", 1), ("G5", 2), ...]
```

### Cellular Automata (Lectures 13 to 14)

A grid where each cell updates according to local rules based on its neighbors. Simple rules produce complex, unpredictable behavior. Mapped to drums: rows = time steps, columns = instruments. A live cell means "play this instrument at this time."

The code uses four musically motivated rules: syncopation resolution (hihat ON + kick OFF → kick ON next beat), gap filling (kick OFF for 2 beats → snare fills in), accenting (kick + snare together → hihat joins), and mutation (random toggles for variation).

Best for percussive or ambient texture generation. Their strength: self similar structures with zero training data.

### Genetic Algorithms (Lectures 15 to 16)

Evolutionary search for music. Start with a population of candidate chord sequences, evaluate each with a fitness function, breed the best via crossover and mutation. Repeat until convergence.

The fitness function for harmonization has four weighted components: chord melody congruence (does the chord contain the melody note?), harmonic flow (do transitions follow common patterns like V→I?), functional harmony (starts on I, ends on I, contains IV and V?), and chord variety. You're searching the space of all possible harmonizations, not enumerating it.

```python
weights = {
    "chord_melody_congruence": 0.4,
    "chord_variety": 0.1,
    "harmonic_flow": 0.3,
    "functional_harmony": 0.2,
}
harmonizer = GeneticMelodyHarmonizer(
    population_size=100, mutation_rate=0.05, ...
)
chords = harmonizer.generate(generations=1000)
```

### Transformers for Music (Lectures 17 to 19)

The dominant architecture. Self attention lets every token attend to every other token simultaneously, no sequential bottleneck. For symbolic music this is huge: bar 1 can directly influence bar 64.

The course uses a simple tokenization: each note is `"pitch-duration"` (e.g. `"C4-1.0"`). The model is an encoder decoder Transformer built in TensorFlow/Keras with sinusoidal positional encoding, multi head attention, and the standard residual + LayerNorm pattern.

Training uses `SparseCategoricalCrossentropy` with mask to ignore padding. Generation is autoregressive greedy decoding: start with a seed sequence, predict the next token, append it, repeat.

```python
generator = MelodyGenerator(transformer_model, tokenizer)
start_sequence = ["C4-1.0", "D4-1.0", "E4-1.0", "C4-1.0"]
new_melody = generator.generate(start_sequence)
```

---

## Part 3: Tools for Musicians (Lectures 20 to 22)

### RAVE: Real Time Audio Synthesis (Lecture 20)

RAVE (Realtime Audio Variational autoEncoder, Caillon et al. 2021) is a neural audio synthesis model that runs in real time. Train it on your own audio corpus (a specific instrument, a soundscape, a voice). It learns that timbral space. Then at inference you explore the learned latent space in real time, or you play an instrument and RAVE morphs the output to sound like your corpus.

```python
model = torch.jit.load("my_corpus.ts")
z = model.encode(audio_tensor)       # (1, latent_dim, T)
z_morphed = z * 1.5                  # stretch the latent
audio_out = model.decode(z_morphed)  # back to waveform
```

The latent space is smooth and musically meaningful: interpolation between two points sounds natural. Works as a Max/MSP or Pure Data external for real time manipulation. Typical training: around 1 hour of audio per model.

### Compose and Embellish: Hierarchical Piano Generation (Lecture 21)

Two stage model: first generate a sparse skeleton (high level harmonic and melodic structure), then embellish it with expressive details (passing notes, ornaments, velocity variation, micro timing). Mirrors how a pianist actually thinks about performance.

Why hierarchical works better than flat note by note generation: separating structure and detail means each model has a simpler task. It also gives the musician a natural intervention point: edit the skeleton, re embellish. Much more intuitive than editing individual MIDI notes from a blob of generated output.

### Mustango: Text to Music with Music Theory Conditioning (Lecture 22)

Mustango (UPF / MTG) is a text to music diffusion model. The differentiator from MusicGen or Suno: it takes music theory aware conditioning. You can specify chord progressions, key, and BPM in your prompt and the model actually respects them.

Architecture: text encoder (FLAN T5 with music theory tokens) → conditioned diffusion on mel spectrogram → HiFi GAN vocoder → audio. Open source, weights and code are public.

Limitations of current text to music: text is a poor interface for music (hard to specify exact notes or structural transitions), generated audio is hard to edit in a DAW (no MIDI, no editable structure), style control is coarse ("jazz" means a thousand things), and quality degrades beyond ~30 seconds.

---

## What I took away

The techniques in this course are the foundations. Grammars, Markov chains, cellular automata, genetic algorithms, Transformers. Newer paradigms (diffusion, flow matching) are becoming dominant. But anyone who has internalized the foundations has the conceptual map to understand whatever comes next. The pattern Valerio keeps making: rule based methods give you control, ML methods give you expressiveness, and the best systems find a way to combine both.

📺 **[Video series](https://www.youtube.com/@ValerioVelwordo/playlists)** · 💻 **[GitHub repo](https://github.com/musikalkemist/generativemusicaicourse)**
