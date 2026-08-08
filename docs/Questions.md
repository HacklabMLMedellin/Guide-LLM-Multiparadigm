# Questions

## 1. How can we measure speed?

How can we measure the speed of a module if the simulator is slower than the real hardware, while the real module could potentially be faster?

**Possible approach:**

```text
Via mathematical formulas and theoretical performance estimates.
```

---

## 2. Valid Libraries for Simulation and Hybrid Algorithms

### Quantum Computing

```text
PennyLane
```

### Thermodynamic Computing

```text
THRML
```

### Photonic Computing for ONN (Optical Neural Networks)

```text
PhotonicTorch
Simphony — Integrated Photonic Circuits
```

### Other Paradigms

```text
Biological Computing
```

---

## 3. What Is the Difference Between Photonic Computing, Quantum Photonic Computing, and Quantum Computing?

| Feature                 | Photonic Computing                              | Quantum Photonic Computing      | Quantum Computing                                            |
| ----------------------- | ----------------------------------------------- | ------------------------------- | ------------------------------------------------------------ |
| **Information carrier** | Classical light (photons)                       | Quantum states of photons       | Qubits (superconductors, trapped ions, photons, atoms, etc.) |
| **Physics**             | Classical electromagnetism                      | Quantum mechanics               | Quantum mechanics                                            |
| **Uses superposition?** | ❌ No                                            | ✅ Yes                           | ✅ Yes                                                        |
| **Uses entanglement?**  | ❌ No                                            | ✅ Yes                           | ✅ Yes                                                        |
| **Uses interference?**  | Classical optical interference                  | Quantum interference            | Quantum interference                                         |
| **Error correction**    | Similar to classical computing                  | Extremely difficult             | Extremely difficult                                          |
| **Goal**                | Faster and more efficient classical computation | Quantum computing using photons | Quantum computing using different qubit technologies         |

---

## 4. How Much Does Each Computer Cost?

### Quantum Computer

```text
Estimated: ~$2 million
```

### Thermodynamic Computer

```text
Price: Not publicly available
```

### Photonic Computer for ONN (Optical Neural Network)

```text
Estimated: €50,000–€100,000
Reference: Q.ANT
```

> **Note:** These are rough estimates and should be verified against current hardware, vendor, and system specifications.

---

## 5. Foundation or Capitalist Company?

```text
We don't know yet.
```

---

# LLM Infrastructure

## 6. Do Models Like ChatGPT, Claude, and Gemini Use Tools Such as Airflow or Databricks?

```text
Probably yes.
```

Large-scale AI companies typically use sophisticated data and ML infrastructure, although the exact internal technology stacks of companies such as OpenAI, Anthropic, and Google are not fully public.

---

## 7. Do They Use Software Engineering Best Practices?

The model is the final product, while the surrounding code is just supporting infrastructure.

```text
They use extensive software engineering practices around the model,
including modularity, testing, distributed training infrastructure,
experiment tracking, data pipelines, monitoring, and reproducibility.
```

The training code is therefore not simply a "trash script." At large scale, it is usually a substantial engineering system.

---

## 8. What Do They Use as a Database, and Why?

For large-scale ML systems, data is often stored in a combination of:

* Object storage
* Data lakes
* Distributed file systems
* Metadata/catalog databases
* Structured datasets such as Parquet

**Parquet is a file format, not a database.**

For example:

```text
Object Storage
      │
      ├── Raw datasets
      ├── Clean datasets
      ├── Deduplicated datasets
      └── Training datasets
                │
                └── Parquet files
```

---

## 9. How Is the Data Prepared?

A typical data preparation pipeline can include:

```text
Extraction
    ↓
Filtering
    ↓
Cleaning
    ↓
Deduplication
    ↓
Quality assessment
    ↓
Normalization
    ↓
Tokenization
    ↓
Dataset creation
    ↓
Training
```

Important steps include:

* **Extraction** — collect data from different sources.
* **Filtering** — remove irrelevant or unwanted content.
* **Cleaning** — remove corrupted or malformed data.
* **Deduplication** — remove duplicate or near-duplicate content.
* **Quality processing** — score and filter data based on quality.
* **Normalization** — standardize formats.
* **Tokenization** — convert text into tokens for model training.

---

## 10. What Does the Data Look Like?

A simplified training-data record could look like:

```json
{
  "text": "Photosynthesis is the process by which...",
  "source": "web",
  "language": "en",
  "quality": 0.94
}
```

A real production dataset would typically contain much more metadata, for example:

```json
{
  "id": "document_001",
  "text": "Photosynthesis is the process by which...",
  "source": "web",
  "language": "en",
  "quality": 0.94,
  "license": "example-license",
  "domain": "science",
  "token_count": 128,
  "deduplication_hash": "abc123..."
}
```
