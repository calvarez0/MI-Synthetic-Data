# Evolutionary Synthetic Data Generation for Motivational Interviewing

An LLM pipeline that generates, evaluates, and curates high-quality synthetic training data for Motivational Interviewing (MI)—specifically cannabis counseling dialogues.

## How It Works

Two local LLMs work together in an evolutionary loop:

```
Generator (WizardLM) ──▶ Evaluator (LLaMA 3) ──▶ Archive (Top-K)
        │                       │                      │
   Creates client          Scores using           Keeps only
   scenarios + MI          MITI 4.2.1            highest-scoring
   therapist responses     clinical criteria      examples
        │                                              │
        └──────────── Diversity feedback ◀─────────────┘
```

1. **Generator** creates realistic client utterances with clinical context, then produces MI-adherent therapist responses
2. **Evaluator** scores each dialogue on realism (1-5) and MI fidelity (1-5) using MITI 4.2.1 criteria
3. **Archive** maintains a quality-ranked collection, discarding lower-scoring examples over time

The key MI constraint: therapist responses must elicit *change talk* ("I want to cut back") and avoid eliciting *sustain talk* ("It helps me relax").

## Files

| File | Description |
|------|-------------|
| `main.py` | Core generation pipeline—run this to generate data |
| `convert_to_alpaca.py` | Converts top examples to Alpaca format for fine-tuning |
| `mi_dialogue_archive.json` | Full archive with all metadata and scores |
| `mi_training_data.json` | Ranked training pairs (top-K by score) |
| `mi_alpaca_format_81.json` | Ready-to-use Alpaca format dataset |
| `cannabis_generation.log` | Terminal output from generation run |

## Usage

Requires [Ollama](https://ollama.ai/) running locally with `wizardlm2:7b` and `llama3.2:latest`.

```bash
# Generate single dialogue
python main.py --mode single

# Run continuous generation
python main.py --mode continuous --target 100

# Convert top 100 to Alpaca format
python convert_to_alpaca.py
```

## Example Output

**Context:** 35-year-old teacher, daily edible use for stress, partner noticed forgetfulness and missed deadlines. Contemplation stage.

**Client:** "I guess I'm here because my partner wanted me to check this out. He thinks I'm using too much. But honestly, it helps me unwind after teaching. I don't think I have a problem—it's more like a routine. But my partner mentioned I've been forgetful, even missing an important deadline. Maybe there's something to this after all."

**Therapist:** "It sounds like you've been grappling with some tough challenges lately. You mentioned your partner's concerns have been on your mind, and you're noticing the pattern of daily use. You've recognized the potential for marijuana to serve as a way to numb out from life's demands—that shows self-awareness. Can you tell me more about what changes you're hoping to see that could align with your values?"

**Scores:** Realism: 4.38 | MI Fidelity: 3.69 | Total: 3.90