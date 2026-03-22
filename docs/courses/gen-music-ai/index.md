# 🎵 Generative Music AI — Course Notes

<p class="hero-subtitle">
Notes from Valerio Velardo's <a href="https://github.com/musikalkemist/generativemusicaicourse">Generative Music AI Course</a> (The Sound of AI, 2023). 22 lectures covering the full spectrum: from rule based symbolic generation to neural audio synthesis. Produced in collaboration with the Music Technology Group at Universitat Pompeu Fabra (UPF) in Barcelona. Three instructors: Valerio Velardo, Iran Roman (NYU), and Xavier Serra (MTG UPF). These notes exist so I can look up a concept in 30 seconds instead of rewatching a video.
</p>

📺 **[Video series](https://www.youtube.com/@ValerioVelwordo/playlists)** · 💻 **[GitHub repo](https://github.com/musikalkemist/generativemusicaicourse)** · Dependencies: `music21==9.9.1`, `tensorflow==2.20.0`

---

## Part 1: Foundations

### Lecture 1 — Course Overview

The structure is intentional: Part 1 builds conceptual foundations, Part 2 goes hands on with algorithms and Python code, Part 3 covers real world tools musicians can pick up today. Each theory lecture has a paired coding lecture.

Setup is minimal. Python 3.8, two dependencies:

```
pip install music21==9.9.1 tensorflow==2.20.0
```

`music21` is MIT's library for music notation and MIDI manipulation. `tensorflow` is used only in the Transformer section. All code is in the [GitHub repo](https://github.com/musikalkemist/generativemusicaicourse), organized per lecture.

### Lecture 2 — What Is Generative Music?

A generative music system is software that automates, to varying degrees, the composition and production of music. The word to focus on is *automates*. The system makes compositional decisions that you would otherwise make manually. The distinction between full automation and co creation is a spectrum, not a binary. Most useful real world systems live somewhere in the middle.

The spectrum of autonomy: traditional composition (fully human) → AI as tool (copilot style, chord suggestions) → co creation (human steers, AI generates) → fully autonomous generation (machine only).

Four dimensions Valerio uses to characterize any system:

**Autonomy**: how much creative control does the system have vs the human? **Interactivity**: can you steer it in real time? **Adaptivity**: does it respond to external signals (game state, emotion, context)? **Originality**: does it recombine existing material or generate genuinely new patterns?

Brian Eno coined "generative music" in 1996. His concept with Music for Airports (1978) was to build a system that never plays the same thing twice. It was a philosophical position before it was a technical one.

### Lecture 3 — History of Generative Music

Five eras. Each unlocked by a new compute paradigm. The pattern repeats every time.

**1700 to 1956, Pre computer era.** Mozart's Musikalisches Würfelspiel (1787): generated minuets by rolling dice. Each die selected a pre composed measure from a lookup table. Controlled randomness, not chaos.

**1957 to 2009, Academic era.** ILLIAC Suite (Hiller and Isaacson, 1957): first computer generated score. Rule based systems, Markov chains, stochastic composition (Xenakis). MAX/MSP born at IRCAM. Everything stays in labs.

**2010 to 2016, First startup wave.** Jukedeck, Amper Music, Melodrive (Valerio's own company). Rule based plus early ML. Commercial focus: music for video, music for games. Market starts paying attention.

**2017 to 2022, Big tech experiments.** Google Magenta, OpenAI MuseNet (GPT 2 fine tuned on music), Jukebox (raw audio generation with VQ VAE). Models get large. Results are surprising. Quality still far from human music.

**2023+, Music AI hype.** MusicLM (Google), MusicGen (Meta), Suno, Udio. Text to music for everyone. Quality jumps sharply. First copyright lawsuits over training data begin.

Every time a new compute infrastructure arrives (mainframes, PCs, cloud, GPU farms, LLMs), generative music makes a step change. It doesn't advance linearly, it jumps.

### Lecture 4 — Use Cases

Four domains where generative music has real commercial traction today.

**Interactive media.** Video games, VR/XR. Soundtrack adapts to game state. Melodrive's approach: system receives emotional signals (tension, calm) and generates music that matches in real time.

**Content creation.** YouTube, podcasts, social media. Thousands of creators need royalty free music daily. Impossible to commission it all. Epidemic Sound, Artlist, and now Suno/Udio operate here.

**Composer tools.** AI as assistant: suggests the next chord progression, completes a melody, harmonizes a line. The composer decides, the AI explores the space.

**Therapeutic / ambient.** Meditation apps, sleep tech, background music for public spaces. Requires continuous music, always slightly different, never jarring. A near perfect use case for generative systems.

When does generative music actually make sense? High volume (too much content to commission manually), adaptivity required (music must respond to context dynamically), the musicality bar is functional not artistic (not an album being critically listened to), and cost sensitivity (licensing or commissioning is prohibitive).

### Lecture 5 — Ethical Implications

Valerio doesn't preach, he maps the tensions. Three main areas.

**Copyright and training data.** Almost all large models were trained on copyrighted music without consent or compensation. The open legal question: is it "fair use" to learn from a work without reproducing it? Who owns generated output? No settled answers yet.

**Labor displacement.** People working in functional music (stock music, library music, social media content) are already affected. Artists making art music are less at risk in the short term because the value is in the human signature, not just the sound. But the displacement is asymmetric: lower barrier to entry jobs go first.

**Attribution and authorship.** If a model generates a melody, who is the author? The model? The person who wrote the prompt? The person who curated the dataset? The person who trained the model? Copyright law was built around the idea of a human author and doesn't know how to answer this yet.

Valerio's framing: AI is no different from introducing music notation, the modern orchestra, or DAWs. Every technology shift displaced roles and created others. The problem isn't the technology itself, it's managing the transition without destroying the people in the middle.

### Lecture 6 — Symbolic vs Audio Generation

This is probably the most important design decision when building a generative system. Do you operate in symbolic space or audio space?

| Dimension | Symbolic (MIDI) | Audio (Waveform) |
|---|---|---|
| Representation | Notes, durations, chords | Raw audio samples or spectrogram |
| Data volume | Small | Very large |
| Interpretability | High, editable in a DAW | Black box output |
| Timbre / expressiveness | Limited | Complete |
| Needs a synthesizer? | Yes, to render | No, direct output |
| Music theory integration | Easy | Hard |
| Examples | MuseNet, Magenta, Music Transformer | RAVE, Jukebox, MusicGen |

The current trend is hybrid: generate symbolic structure with a Transformer, then synthesize with a neural audio model like RAVE. Best of both worlds, at the cost of complexity.

### Lecture 7 — Generative Techniques Overview

A map of the territory. Everything in Part 2 lives somewhere in this taxonomy.

**Rule based / Symbolic:** Generative Grammars, Markov Chains, Cellular Automata, Genetic Algorithms.

**Machine Learning:** RNN/LSTM, Transformers, VAE/GAN, Diffusion.

**Audio Specific:** WaveNet, RAVE, AudioLDM.

The key tension: rule based methods are controllable and interpretable but require hand crafted knowledge. ML methods learn complex patterns from data but are hard to steer. Hybrid approaches (grammar + neural) try to get both. That's where active research is happening.

### Lecture 8 — Limitations and Future Vision

This is the most useful lecture in Part 1 because it gives you the tools to evaluate any system you encounter. Valerio stops being the enthusiastic educator and gets critical.

**What current systems still can't do well:**

**Long range coherence:** systems generate locally convincing music but lose global structure (verse/chorus, development, tension/release arcs). No model today can reliably build a compelling 5 minute musical arc.

**Music theory ignorance:** most models learn statistics, not harmonic logic. They can violate basic rules (parallel fifths, voice crossing) that no composition student would make.

**Controllability:** saying "make it more tense here, then resolve there" is still very hard. Systems respond to coarse text prompts, not precise musical instructions.

**Evaluation:** there is no good objective metric for musical quality. Subjective listening tests don't scale. Perplexity and other ML proxies correlate poorly with perceived quality.

**Where the field is heading:** neuro symbolic integration (combining ML with explicit music theory rules), multimodal generation (music conditioned on video, emotion, narrative), interactive co creation (AI as real time collaborator in live performance), and personalization (systems that learn a specific musician's style).

---

## Part 2: Technical Dive

### Lecture 9 — Generative Grammars

A formal grammar defines rewrite rules that expand non terminal symbols into sequences. Apply it to music: "Phrase" expands to "Motif + Cadence", "Motif" expands to specific note patterns. Essentially Chomsky applied to composition.

```
Formal grammar: G = (V, Σ, R, S)

V  = non-terminal symbols   (Phrase, Motif, Cadence ...)
Σ  = terminal symbols       (actual notes: C4, D4, E4 ...)
R  = production rules       (Phrase → Motif Cadence)
S  = start symbol           (root of the expansion)
```

**L Systems.** Invented by Lindenmayer for modelling plant growth. Mapped to music: symbols correspond to pitches and durations. They produce self similar, fractal like melodies.

```
L-System example:
Axiom: A
Rules: A → ABC, B → BA, C → EF, F → GFD

After 4 iterations:
A → ABC → ABCBAEF → ABCBAEFBAABCGEFDDAEF → ...

Each letter maps to a chord (C = Cmaj, D = Dmin, G = Gmaj ...)
Result: a self-similar chord progression that grows fractally
```

### Lecture 10 — Chord Generation with L Systems (Code)

The actual code from `lsystem.py` in the repo. A clean `LSystem` class, then maps the output string to music21 chords. Each letter in the output string corresponds to a diatonic chord in C major.

```python
class LSystem:
    def __init__(self, axiom, rules):
        self.axiom = axiom
        self.rules = rules
        self.output = axiom

    def iterate(self, n=1):
        for i in range(n):
            next_output = self._iterate_once()
            self.output = next_output
            print(f"Output after {i + 1} iteration(s): {self.output}")
        final_output = self.output
        self._reset_output()
        return final_output

    def _iterate_once(self):
        symbols = [self._apply_rule(symbol) for symbol in self.output]
        return "".join(symbols)

    def _apply_rule(self, symbol):
        return self.rules.get(symbol, symbol)
```

Mapping to music21 chords:

```python
def l_system_to_music21_chords(chord_sequence):
    chord_dict = {
        "C": ["C", "E", "G"],   # Cmaj
        "D": ["D", "F", "A"],   # Dmin
        "E": ["E", "G", "B"],   # Emin
        "F": ["F", "A", "C"],   # Fmaj
        "G": ["G", "B", "D"],   # Gmaj
        "A": ["A", "C", "E"],   # Amin
        "B": ["B", "D", "F"],   # Bdim
    }
    return [chord.Chord(chord_dict[c]) for c in chord_sequence if c in chord_dict]

# Usage:
axiom = "A"
rules = {"A": "ABC", "B": "BA", "C": "EF", "F": "GFD"}
l_system = LSystem(axiom, rules)
chord_sequence = l_system.iterate(4)
music21_chords = l_system_to_music21_chords(chord_sequence)
```

### Lecture 11 — Markov Chains

A Markov chain is a probabilistic model where the next state depends *only on the current state*, not the entire history. This is the Markov property. Applied to melody: given the note I'm playing right now, what note is most likely next?

The key difference from grammars: the probabilities are *learned from data* instead of hand written. Give it a corpus of MIDI melodies and it builds the transition matrix automatically.

```
The Markov property (order 1):
P(X_{t+1} = x | X_t, X_{t-1}, ..., X_1) = P(X_{t+1} = x | X_t)

The distribution of the next state depends only on the current state.

For order-n chains:
P(X_{t+1} = x | X_t, X_{t-1}, ..., X_{t-n}) = p

Order 1: depends on last note only
Order 2: depends on last 2 notes
Order n: more context, less variety, more memorization
```

The transition matrix `T[i][j] = P(next=j | current=i)`: each row sums to 1. The highest value in each row is the most probable next note from that state.

### Lecture 12 — Melody Generation with Markov Chains (Code)

The actual code from `markovchain.py`. The state is a `(pitch, duration)` tuple, so the chain models both note identity and rhythmic value simultaneously. The training data is "Twinkle Twinkle Little Star".

```python
class MarkovChainMelodyGenerator:
    def __init__(self, states):
        self.states = states
        self.initial_probabilities = np.zeros(len(states))
        self.transition_matrix = np.zeros((len(states), len(states)))
        self._state_indexes = {state: i for (i, state) in enumerate(states)}

    def train(self, notes):
        # notes: list of music21.note.Note objects
        self._calculate_initial_probabilities(notes)
        self._calculate_transition_matrix(notes)

    def generate(self, length):
        melody = [self._generate_starting_state()]
        for _ in range(1, length):
            melody.append(self._generate_next_state(melody[-1]))
        return melody

    def _normalize_transition_matrix(self):
        row_sums = self.transition_matrix.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.transition_matrix = np.where(
                row_sums[:, None],
                self.transition_matrix / row_sums[:, None],
                0,   # keep zeros where sum is 0
            )

    def _generate_next_state(self, current_state):
        if self._does_state_have_subsequent(current_state):
            index = np.random.choice(
                list(self._state_indexes.values()),
                p=self.transition_matrix[self._state_indexes[current_state]],
            )
            return self.states[index]
        return self._generate_starting_state()   # fallback
```

Training data and usage:

```python
def create_training_data():
    # "Twinkle Twinkle Little Star" as music21 Note objects
    return [
        note.Note("C5", quarterLength=1), note.Note("C5", quarterLength=1),
        note.Note("G5", quarterLength=1), note.Note("G5", quarterLength=1),
        note.Note("A5", quarterLength=1), note.Note("A5", quarterLength=1),
        note.Note("G5", quarterLength=2),   # Twinkle, twinkle, little star
        # ... etc
    ]

states = [("C5", 1), ("D5", 1), ("E5", 1), ("F5", 1), ("G5", 1),
          ("A5", 1), ("C5", 2), ("G5", 2), ("A5", 2)]
model = MarkovChainMelodyGenerator(states)
model.train(create_training_data())
generated_melody = model.generate(40)
visualize_melody(generated_melody)   # opens music21 score viewer
```

Each state is a `(pitch, duration)` pair, not just a pitch. This means the chain also learns rhythmic patterns. If C5 quarter note tends to follow G5 quarter note, it captures that. Modeling both dimensions at once is one of the cleaner design choices in this codebase.

### Lecture 13 — Cellular Automata

A cellular automaton is a grid system where each cell has a state (0/1) and updates at each time step according to local rules based on its neighbors. The surprising thing: simple rules produce complex, often unpredictable behavior.

```
Wolfram Rule 30 (1D CA):
Neighborhood: [left, center, right] → new center value

111→0  110→0  101→0  100→1
011→1  010→1  001→1  000→0

"30" is the rule in binary: 00011110 = 30

Evolution from a single 1:
t=0: ...0001000...
t=1: ...0011100...
t=2: ...0110010...
t=3: ...1101111...
Chaotic, aperiodic — Wolfram uses it as a random number generator
```

**Mapping a CA to music:** Rows = time steps, columns = pitches or instruments. A "live" (1) cell in column C at time T means: play note C at time T. Different rules produce different rhythmic and melodic densities.

CAs produce mechanical, pattern heavy music. Best for percussive or ambient texture content, not expressive melody. Their strength is generating self similar structures with zero training data.

### Lecture 14 — Drum Generation with Cellular Automata (Code)

The code from `cellularautomaton.py`. More sophisticated than a basic Wolfram CA: four musical rules govern how the drum state evolves. They map to musical concepts: syncopation, gap filling, accenting, and random mutation.

```python
class CellularAutomatonDrumGenerator:
    HIHAT_ON_PROBABILITY = 0.7
    MUTATION_PROBABILITY = 0.1

    def __init__(self, pattern_length):
        self.pattern_length = pattern_length
        self.state = self._initialize_state(pattern_length)
        self._rules = {
            "syncopation_resolution": self._apply_syncopation_resolution_rule,
            "filling_gaps":          self._apply_filling_gaps_rule,
            "accenting":             self._apply_accenting_rule,
            "mutation":              self._apply_mutation_rule,
        }

    def _apply_syncopation_resolution_rule(self, position, new_state):
        # If hihat ON and kick OFF at position → turn kick ON at next position
        next_pos = self._get_next_position(position)
        if (self.state[DrumInstruments.HIHAT.value][position] == DrumStates.ON.value
                and self.state[DrumInstruments.KICK.value][position] == DrumStates.OFF.value):
            new_state[DrumInstruments.KICK.value][next_pos] = DrumStates.ON.value
        return new_state

    def _apply_filling_gaps_rule(self, position, new_state):
        # If kick OFF for 2 consecutive positions → turn snare ON next position
        prev_pos = self._get_previous_position(position)
        next_pos = self._get_next_position(position)
        if (self.state[DrumInstruments.KICK.value][prev_pos] == DrumStates.OFF.value
                and self.state[DrumInstruments.KICK.value][position] == DrumStates.OFF.value):
            new_state[DrumInstruments.SNARE.value][next_pos] = DrumStates.ON.value
        return new_state

    def _apply_accenting_rule(self, position, new_state):
        # If kick AND snare both ON → turn hihat ON with HIHAT_ON_PROBABILITY
        if (self.state[DrumInstruments.KICK.value][position] == DrumStates.ON.value
                and self.state[DrumInstruments.SNARE.value][position] == DrumStates.ON.value):
            state = DrumStates.ON.value if random.random() < self.HIHAT_ON_PROBABILITY else DrumStates.OFF.value
            new_state[DrumInstruments.HIHAT.value][position] = state
        return new_state

    def _apply_mutation_rule(self, position, new_state):
        # With MUTATION_PROBABILITY: randomly toggle a random instrument
        if random.random() < self.MUTATION_PROBABILITY:
            inst  = random.choice([DrumInstruments.KICK, DrumInstruments.SNARE, DrumInstruments.HIHAT])
            state = random.choice([DrumStates.ON.value, DrumStates.OFF.value])
            new_state[inst.value][position] = state
        return new_state
```

Usage:

```python
drum_generator = CellularAutomatonDrumGenerator(pattern_length=16)
music_converter = DrumPatternMusic21Converter()

for _ in range(8):
    drum_generator.step()          # evolve the CA 8 times

score = music_converter.to_music21_score(drum_generator.state)
score.show()
```

The MIDI pitches used are standard General MIDI drum mappings: Kick = 36, Snare = 38, Hi Hat = 42. The converter creates a separate music21 `Part` per instrument and fills it with `Note`/`Rest` objects.

### Lecture 15 — Genetic Algorithms

Genetic algorithms are inspired by natural selection. Start with a population of candidate solutions (chord sequences, harmonizations), evaluate each with a fitness function, breed the best via crossover and mutation. Repeat until convergence.

The key insight: you're not enumerating the space of all possible chord sequences (impossible). You're *searching* that space guided by musical quality criteria.

The loop: Initialize (N random candidates) → Evaluate (fitness f(x) per candidate) → Select (fittest, via roulette wheel) → Crossover (combine 2 parents) → Mutate (small random changes) → back to Evaluate. Repeats until fitness threshold or max generations.

### Lecture 16 — Melody Harmonization with Genetic Algorithms (Code)

Given a fixed melody (soprano), the GA finds the optimal chord progression to harmonize it. The chromosome is a sequence of chord choices, one per bar. This is `geneticmelodyharmonizer.py`, the most complex file in the repo.

The fitness function has four components with explicit weights:

```python
class FitnessEvaluator:
    def evaluate(self, chord_sequence):
        # weighted sum of four musical criteria
        return sum(
            self.weights[func] * getattr(self, f"_{func}")(chord_sequence)
            for func in self.weights
        )

    def _chord_melody_congruence(self, chord_sequence):
        # Does each chord contain the melody note at that beat?
        score, melody_index = 0, 0
        for chord in chord_sequence:
            bar_duration = 0
            while bar_duration < 4 and melody_index < len(self.melody_data.notes):
                pitch, duration = self.melody_data.notes[melody_index]
                if pitch[0] in self.chord_mappings[chord]:
                    score += duration
                bar_duration += duration
                melody_index += 1
        return score / self.melody_data.duration

    def _harmonic_flow(self, chord_sequence):
        # How often do transitions match preferred_transitions?
        score = sum(1 for i in range(len(chord_sequence) - 1)
                    if chord_sequence[i+1] in self.preferred_transitions[chord_sequence[i]])
        return score / (len(chord_sequence) - 1)

    def _functional_harmony(self, chord_sequence):
        # Starts with I or vi, ends with I, contains IV and V?
        score = 0
        if chord_sequence[0] in ["C", "Am"]: score += 1
        if chord_sequence[-1] in ["C"]:      score += 1
        if "F" in chord_sequence and "G" in chord_sequence: score += 1
        return score / 3
```

Weights and preferred transitions:

```python
weights = {
    "chord_melody_congruence": 0.4,
    "chord_variety":           0.1,
    "harmonic_flow":           0.3,
    "functional_harmony":      0.2,
}
preferred_transitions = {
    "C":    ["G", "Am", "F"],
    "Dm":   ["G", "Am"],
    "Em":   ["Am", "F", "C"],
    "F":    ["C", "G"],
    "G":    ["Am", "C"],
    "Am":   ["Dm", "Em", "F", "C"],
    "Bdim": ["F", "Am"],
}
harmonizer = GeneticMelodyHarmonizer(
    melody_data=melody_data, chords=list(chord_mappings.keys()),
    population_size=100, mutation_rate=0.05,
    fitness_evaluator=fitness_evaluator,
)
generated_chords = harmonizer.generate(generations=1000)
```

### Lecture 17 — Transformers Part 1: The Architecture

The Transformer is the dominant architecture in modern generative AI, for both text (GPT) and music. The central mechanism is self attention: every token in the sequence can attend to every other token simultaneously, with no sequential bottleneck like RNNs had. For symbolic music this is a big deal: bar 1 can directly influence bar 64.

**Scaled Dot Product Attention (Vaswani et al., 2017):**

```
Attention(Q, K, V) = softmax( Q · Kᵀ / √dₖ ) · V

Q = Query matrix    "what am I looking for?"
K = Key matrix      "what do I have to offer?"
V = Value matrix    "what do I pass forward if chosen?"

Each token produces Q, K, V by projecting its embedding through learned weight matrices.
QKᵀ scores how much each pair of tokens should attend to each other.
Divide by √dₖ to prevent vanishing gradients as dₖ grows.
softmax → probability distribution over all positions.
Output = weighted average of Values.
```

**Multi Head Attention:**

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) · Wᴼ

where  headᵢ = Attention(Q·Wᵢᴼ, K·Wᵢᴷ, V·Wᵢᵛ)

h different heads learn to attend to different aspects:
  one head might learn rhythmic relationships
  another harmonic relationships
  another long-range motivic structure
```

**Sinusoidal positional encoding.** Since attention is permutation invariant (it doesn't care about order), you need to inject position information:

```python
def sinusoidal_position_encoding(num_positions, d_model):
    positions = np.arange(num_positions)[:, np.newaxis]   # (T, 1)
    dims = np.arange(d_model)[np.newaxis, :]               # (1, d_model)

    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / d_model)
    angles = positions * angle_rates

    pos_encoding = np.zeros_like(angles)
    pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])   # sin for even dims
    pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])   # cos for odd dims

    return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)
```

### Lecture 18 — Transformers Part 2: Music Tokenization

How do you turn music into tokens? The course uses a simple string representation: each note is encoded as `"pitch-duration"`, e.g. `"C4-1.0"` for a C4 quarter note. This goes into a Keras tokenizer. Simple but effective for the tutorial scope.

The full EncoderLayer (one transformer block):

```python
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_feedforward, dropout_rate=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(key_dim=d_model, num_heads=num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(d_feedforward, activation="relu"),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, training, mask):
        attn_output = self.mha(x, x, x, attention_mask=mask)   # self-attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)                # residual connection

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)              # residual connection
        return out2
```

Model instantiation:

```python
transformer_model = Transformer(
    num_layers=2,
    d_model=64,
    num_heads=2,
    d_feedforward=128,
    input_vocab_size=vocab_size,
    target_vocab_size=vocab_size,
    max_num_positions_in_pe_encoder=100,
    max_num_positions_in_pe_decoder=100,
    dropout_rate=0.1,
)
```

This is an encoder decoder Transformer (not decoder only like GPT). For music generation you'd typically use decoder only. The encoder decoder design here is a pedagogical choice: it shows both components. The masking (look ahead and padding) is explicitly left as an exercise.

### Lecture 19 — Melody Generation with Transformers (Code)

The full pipeline: preprocess MIDI to a JSON dataset, train the Transformer, generate melodies autoregressively. Training uses `SparseCategoricalCrossentropy` and Adam. Generation uses greedy decoding (always picks the highest scoring token).

Training loop:

```python
optimizer = Adam()
sparse_categorical_crossentropy = SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)

@tf.function
def _train_step(input, target, transformer):
    target_input = _right_pad_sequence_once(target[:, :-1])  # all but last
    target_real  = _right_pad_sequence_once(target[:,  1:])  # all but first

    with tf.GradientTape() as tape:
        predictions = transformer(input, target_input, training=True,
                                  enc_padding_mask=None,
                                  look_ahead_mask=None,
                                  dec_padding_mask=None)
        loss = _calculate_loss(target_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    return loss

def _calculate_loss(real, pred):
    loss_ = sparse_categorical_crossentropy(real, pred)
    # mask out padded positions (value 0) from the loss
    mask = tf.cast(tf.math.logical_not(tf.math.equal(real, 0)), dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)
```

Autoregressive generation:

```python
class MelodyGenerator:
    def generate(self, start_sequence):
        input_tensor = self._get_input_tensor(start_sequence)
        num_to_generate = self.max_length - len(input_tensor[0])

        for _ in range(num_to_generate):
            predictions = self.transformer(input_tensor, input_tensor,
                                           training=False,
                                           enc_padding_mask=None,
                                           look_ahead_mask=None,
                                           dec_padding_mask=None)
            # greedy: take the argmax at the last position
            predicted_note = tf.argmax(predictions[:, -1, :], axis=1).numpy()[0]
            input_tensor = tf.concat([input_tensor, [[predicted_note]]], axis=-1)

        return self.tokenizer.sequences_to_texts(input_tensor.numpy())[0]

# Usage:
generator = MelodyGenerator(transformer_model, melody_preprocessor.tokenizer)
start_sequence = ["C4-1.0", "D4-1.0", "E4-1.0", "C4-1.0"]
new_melody = generator.generate(start_sequence)
```

---

## Part 3: Tools for Musicians

### Lecture 20 — Audio Generation with RAVE

RAVE (Realtime Audio Variational autoEncoder, Caillon et al. 2021) is an open source neural audio synthesis model that runs in real time. That last part matters: you can use it in a live performance context, not just in the studio.

The idea: train RAVE on your own audio corpus (a specific instrument, a soundscape, a voice). It learns that timbral space. Then at inference you explore the learned latent space in real time, or you play an instrument and RAVE morphs the output to sound like your corpus.

Architecture: CNN encoder (multiscale strided convolutions + reparameterization) compresses waveform into latent space z, then CNN decoder (with adversarial discriminator loss, GAN style) reconstructs audio. At inference, manipulate z to explore timbre.

```python
# 1. Train (1-3h on GPU with ~1h of audio)
# rave train --config v2 --db_path ./dataset --name my_corpus --out_path ./runs

# 2. Export as torchscript for Max/MSP or Python use
# rave export --run ./runs/my_corpus --streaming

# 3. Use in Python
import torch

model = torch.jit.load("my_corpus.ts")
model.eval()

with torch.no_grad():
    z = model.encode(audio_tensor)           # shape: (1, latent_dim, T)
    z_morphed = z * 1.5                      # stretch the latent
    audio_out = model.decode(z_morphed)      # back to waveform
```

Key properties: the latent space is smooth and musically meaningful (interpolation between two points sounds natural, not glitchy), works as a Max/MSP or Pure Data external for real time manipulation inside a patch, enables timbre transfer (play any instrument, output sounds like the trained corpus), and typical training data is around 1 hour of audio per model.

### Lecture 21 — Piano Generation with Compose and Embellish

Compose and Embellish is a two stage model for piano music generation: first generate a sparse "skeleton" (high level harmonic and melodic structure), then embellish it with expressive details. It mirrors how a pianist actually thinks about performance.

**Stage 1: COMPOSE.** Transformer on high level events. Output: sparse skeleton (main notes + harmony).

**Stage 2: EMBELLISH.** Transformer conditioned on the skeleton. Output: dense piano roll with dynamics + timing.

What Embellish adds on top of the skeleton: passing notes, ornaments, and trills. Velocity variation (dynamics: piano, forte, crescendo). Micro timing deviations (humanization). Arpeggios and harmonic infill.

Why hierarchical generation works better than flat note by note: flat generation loses global structure quickly. Separating structure and detail means each model has a simpler task. It also gives the musician a natural intervention point: edit the skeleton, re embellish. That's a much more intuitive creative workflow than editing individual MIDI notes from a blob of generated output.

### Lecture 22 — Text to Music Generation with Mustango

Mustango is a text to music diffusion model developed by the Music Technology Group at UPF, the same institution running this course. The key differentiator from MusicGen or Suno: it takes music theory aware conditioning. You can specify chord progressions, key, and BPM in your prompt and the model actually respects them.

Architecture: text prompt → Text Encoder (FLAN T5 with music theory tokens) → Diffusion Model (iterative denoising from Gaussian noise on mel spectrogram, conditioned on text) → Vocoder (HiFi GAN, mel to waveform) → audio output.

**What sets Mustango apart:** music theory conditioning (explicitly specify chord progressions, key, BPM and the model respects them, unlike MusicGen and Suno which interpret such prompts loosely), open source (weights and code are public, you can fine tune on your own corpus), and it's a research tool (interpretable conditioning makes it useful for studying controllable generation).

**Limitations of current text to music:** text is a poor interface for music (hard to specify exact note events or structural transitions), generated audio is difficult to edit in a DAW (no MIDI, no editable structure), style control is coarse ("jazz" means a thousand different things), and quality degrades significantly beyond ~30 seconds.

Valerio's closing point: the field moves fast. The techniques in this course are the foundations: grammars, Markov chains, CAs, GAs, Transformers. Newer paradigms (diffusion, flow matching) are becoming dominant. But anyone who has internalized the foundations here has the conceptual map to understand whatever comes next.

---

## What I took away

The pattern Valerio keeps making: rule based methods give you control, ML methods give you expressiveness, and the best systems find a way to combine both. The biggest open problems are long range coherence (nobody can reliably generate a compelling 5 minute musical arc), controllability (coarse text prompts vs precise musical instructions), and evaluation (no good objective metric for musical quality). Every technique in Part 2 will show up again in modern systems, just at a larger scale.

📺 **[Video series](https://www.youtube.com/@ValerioVelwordo/playlists)** · 💻 **[GitHub repo](https://github.com/musikalkemist/generativemusicaicourse)**
