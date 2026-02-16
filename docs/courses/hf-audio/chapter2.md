# Chapter 2: Audio Applications with Pipelines

<p class="hero-subtitle">
Following the <a href="https://huggingface.co/learn/audio-course/en/chapter2/introduction">Hugging Face Audio Course, Unit 2</a>. The course shows three tasks: audio classification, ASR, and audio generation. I adapt all three to environmental sounds and soundscapes instead of speech. The key idea of this chapter: you can do a lot with pre-trained models and zero training.
</p>

📓 **[Full notebook on GitHub](https://github.com/my-sonicase/learn-gen-AI-audio/blob/main/notebooks/chapter2_audio_applications.ipynb)**

| HF Course | My version |
|---|---|
| Audio Classification on MINDS 14 (speech intent) | Audio Classification on **ESC 50** (environmental sounds) |
| ASR with Whisper (speech to text) | **Audio Captioning** (sound to description) |
| TTS with Bark + Music with MusicGen | **Soundscape generation** with MusicGen |

---

## Part 1: Audio Classification

Audio classification = give the model an audio clip, get back a label (or a ranked list of labels with scores). The HF course uses a model for speech intent classification. We use the **Audio Spectrogram Transformer (AST)** by MIT, trained on AudioSet (527 sound event classes).

```python
from transformers import pipeline

classifier = pipeline(
    "audio-classification",
    model="MIT/ast-finetuned-audioset-10-10-0.4593",
)
```

### Classify ESC 50 examples

Let's test it on sounds from the ESC 50 dataset. The model was trained on AudioSet which has different label names, so they won't match exactly. But the predictions should be semantically close.

```python
test_categories = ["rain", "thunderstorm", "church_bells", "sea_waves", "dog"]

for cat in test_categories:
    example = dataset.filter(lambda x: x["category"] == cat)[0]
    audio_array = np.array(example["audio"]["array"], dtype=np.float32)
    result = classifier(audio_array)
```

| True label | Top 1 prediction | Score | Top 2 | Score | Top 3 | Score |
|---|---|---|---|---|---|---|
| rain | Rain | 0.396 | Rain on surface | 0.352 | Raindrop | 0.191 |
| thunderstorm | Thunder | 0.548 | Thunderstorm | 0.254 | Rain | 0.109 |
| church_bells | Church bell | 0.721 | Bell | 0.234 | Change ringing | 0.024 |
| sea_waves | Waves, surf | 0.415 | Ocean | 0.310 | Wind | 0.058 |
| dog | Bark | 0.192 | Animal | 0.184 | Dog | 0.108 |

Pretty solid. The model has never seen ESC 50 data, yet it nails the semantics. Church bells is the most confident prediction. Dog is the weakest, probably because AudioSet has many fine grained animal categories.

### Classify our own sounds

Now our thunder and chimes recordings. The model has never seen these specific files.

| Sound | Top 1 | Score | Top 2 | Score | Top 3 | Score |
|---|---|---|---|---|---|---|
| Thunder/Rain | Thunder | 0.446 | Thunderstorm | 0.424 | Rain | 0.118 |
| Chimes | Wind chime | 0.561 | Chime | 0.390 | Tubular bells | 0.016 |

Both nailed. For thunder it can't decide between Thunder and Thunderstorm (makes sense, the recording has both). For chimes, Wind chime is the top prediction, which is exactly right.

### Under the hood

The `pipeline()` is convenient but hides what's happening. Here's the manual version:

```python
from transformers import AutoFeatureExtractor, ASTForAudioClassification

model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
model = ASTForAudioClassification.from_pretrained(model_id)

# Preprocess
clip = thunder[:SR * 10]
inputs = feature_extractor(clip, sampling_rate=SR, return_tensors="pt")
# Input shape: (1, 1024, 128)

# Inference
with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
```

That's exactly what `pipeline()` does in one line. Good to understand, but use the pipeline in practice.

---

## Part 2: Audio Captioning (instead of ASR)

The HF course uses Whisper for ASR (speech to text). Since our sounds are not speech, we use a Whisper model fine tuned for **audio captioning**: it generates a free text description of what it hears.

The model (`MU-NLPC/whisper-tiny-audio-captioning`) is a standard Whisper encoder decoder fine tuned on audio captioning data. The authors provide a custom model class, but it breaks with recent `transformers` versions. No problem: the weights are identical, so we load them with `WhisperForConditionalGeneration` and handle the style prefix ourselves.

This is a useful lesson: **when a custom class breaks, understand what it does and replicate it with standard tools.**

```python
from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor

checkpoint = "MU-NLPC/whisper-tiny-audio-captioning"
captioning_model = WhisperForConditionalGeneration.from_pretrained(checkpoint)
tokenizer = WhisperTokenizer.from_pretrained(checkpoint)
cap_feature_extractor = WhisperFeatureExtractor.from_pretrained(checkpoint)
```

The model supports 3 captioning styles via a text prefix:

```python
def caption_audio(audio_array, sr, style="clotho > caption: "):
    features = cap_feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt")
    prefix_ids = tokenizer(style, return_tensors="pt").input_ids
    generated = captioning_model.generate(
        features.input_features,
        decoder_input_ids=prefix_ids
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True)
```

### Caption our own sounds

| Sound | Style | Caption |
|---|---|---|
| Thunder/Rain | Natural (Clotho) | A man is walking down the street with thunder in his hands |
| Thunder/Rain | Short (AudioCaps) | Rain falling and th falling |
| Thunder/Rain | Keywords (AudioSet) | natural, fire, natural, natural |
| Chimes | Natural (Clotho) | A person is tapping a glass glass in a room |
| Chimes | Short (AudioCaps) | A series of musical tones playing in a musical instrument |
| Chimes | Keywords (AudioSet) | onomatopoeia, jingle, alarm |

The natural caption for thunder is hilariously wrong ("a man walking down the street with thunder in his hands") but the short caption and keywords get the gist. This is the **tiny** model (39M parameters). Larger variants produce significantly better captions but don't fit in Colab free tier.

### Caption ESC 50 examples

| True label | Generated caption |
|---|---|
| rain | A large volume of water splashes as it flows |
| thunderstorm | Thunder rumbling in the distance |
| sea_waves | A large body of water |
| church_bells | Bells ringing in a church |
| dog | A dog barking |

Much better on the clean 5 second clips. The thunderstorm and church bells captions are surprisingly good for a tiny model.

### ASR vs Audio Captioning

| | ASR (Speech to Text) | Audio Captioning |
|---|---|---|
| **Input** | Speech audio | Any audio |
| **Output** | Exact transcription of words | Free text description of sounds |
| **Architecture** | Encoder decoder (Whisper) | Same encoder decoder (fine tuned Whisper) |
| **Training data** | Speech + transcription pairs | Audio + description pairs |
| **Use case** | Subtitles, dictation, voice assistants | Accessibility, search, metadata generation |

Same architecture, different training data, completely different task. That's the power of transfer learning.

---

## Part 3: Audio Generation with MusicGen

The HF course shows text to speech with Bark and music generation with MusicGen. Since we're doing soundscapes, we use **MusicGen** to generate environmental audio from text prompts.

```python
music_pipe = pipeline(
    "text-to-audio",
    model="facebook/musicgen-small",
    device="cuda" if torch.cuda.is_available() else "cpu",
)
```

### Generate soundscapes from text

Three prompts, three generated sounds:

**"Soft rain falling in a tropical forest with distant thunder and crickets"**
<audio controls>
  <source src="audio/gen_rain_forest.mp3" type="audio/mpeg">
</audio>

**"Gentle wind chimes ringing with different tones in a peaceful garden"**
<audio controls>
  <source src="audio/gen_wind_chimes.mp3" type="audio/mpeg">
</audio>

**"Ocean waves crashing on a rocky shore with seagulls"**
<audio controls>
  <source src="audio/gen_ocean_waves.mp3" type="audio/mpeg">
</audio>

MusicGen is primarily a music model, so it tends to interpret environmental prompts as "music inspired by" rather than literal soundscapes. Still interesting to hear how it translates text to audio.

### Real recording vs generated

How does our real thunder/rain field recording compare to what MusicGen generates from a similar prompt?

**Real thunder/rain (first 10s):**
<audio controls>
  <source src="audio/real_thunder_10s.mp3" type="audio/mpeg">
</audio>

**Generated: "Soft rain falling in a tropical forest with distant thunder"**
<audio controls>
  <source src="audio/gen_thunder_compare.mp3" type="audio/mpeg">
</audio>

You can hear the difference immediately. The real recording has all the messy complexity of nature: layered rain, cicadas, irregular drips. The generated version is smoother, more "musical." This gap is exactly what models like AudioLDM and Stable Audio are trying to close.

### The full loop: Audio → Text → Audio

Here's something fun. Take a recording, caption it, then generate new audio from that caption. A full Audio → Text → Audio round trip.

```python
clip = thunder[:SR * 30]
caption = caption_audio(clip, SR, style="audiocaps > caption: ")
# → "Rain falling and th falling"
output = music_pipe(caption, forward_params={"max_new_tokens": 512})
```

**Original recording:**
<audio controls>
  <source src="audio/loop_original.mp3" type="audio/mpeg">
</audio>

**Caption generated:** "Rain falling and th falling"

**Regenerated from caption:**
<audio controls>
  <source src="audio/loop_regenerated.mp3" type="audio/mpeg">
</audio>

Information is lost at each step. The caption doesn't capture everything in the original, and MusicGen interprets the caption in its own way. Still, it's remarkable that this works at all with pre trained models and zero custom training.

---

## What I learned

Three `pipeline()` tasks, zero training, and we got surprisingly far:

| Task | Model | What it does |
|---|---|---|
| Audio Classification | `MIT/ast-finetuned-audioset` | Audio → label (e.g. "rain", "chime") |
| Audio Captioning | `MU-NLPC/whisper-tiny-audio-captioning` | Audio → free text description |
| Audio Generation | `facebook/musicgen-small` | Text → audio |

`pipeline()` abstracts away all preprocessing. You give it raw audio, it gives you results. When there's no pipeline wrapper (like the captioning model), you load model, feature extractor, and tokenizer separately. Pre trained models get you surprisingly far, but they have blind spots. Fine tuning on your specific data (coming in later chapters) is what fixes that. And you can chain tasks together (caption → generate) for creative applications, even if information is lost at each step.

📓 **[Full notebook with all the code](https://github.com/my-sonicase/learn-gen-AI-audio/blob/main/notebooks/chapter2_audio_applications.ipynb)**
