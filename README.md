# InsightHub-Models – Image→Music Multimodal Retrieval  
### *Research Branch Overview (CLIP×CLAP · Text-Pivot · Emo-CLIP · ImageBind)*

InsightHub is a research-driven project that benchmarks **multiple multimodal alignment strategies** to determine the **best combination of models and methods** for recommending music from user-uploaded images.  
We systematically compare different cross-modal approaches to understand how image mood, theme, semantics, and motion cues map to musical characteristics.

This repository hosts independent research branches—each representing a different alignment philosophy—allowing us to evaluate and combine models to discover the **optimal retrieval pipeline for image-driven music recommendation**.

Following the research phase, these findings will be used to build a **production-ready SNS product** that recommends music based on the aesthetic, mood, and meaning of images posted by users.

---

## Cross-Modal Alignment Dimensions

We evaluate all models across the following dimensions:

- **Cross-modal synchrony**  
  - Energy coherence  
  - Temporal rhythm ↔ visual motion  
  - Color–timbre association  
- **Semantic match**
- **Thematic match**
- **Affective (emotional) alignment**

Each research branch is intentionally chosen as a representative model for one primary dimension, enabling clean and interpretable comparisons across alignment types.

---

# Branch Overview  
Below is a high-level summary of each research branch and its representative role.  
(Full pipelines and implementation details are provided inside each branch directory.)

---

## 1. `text-pivot/` — **Representative of *Thematic Match***
### Concept  
Normalize both images and music into the **text domain**, then match based on shared concepts/themes.

### Why This Branch Matters  
This approach excels at reasoning about **topics, context, genres, and themes** using captions, hashtags, lyrics, and metadata.  
It serves as the **thematic alignment** representative.

### Alignment Focus  
- **Thematic match** ⭐⭐⭐⭐  
- Semantic match ⭐⭐⭐⭐⭐  

---

## 2. `clip-x-clap/` — **Representative of *Semantic Match***
### Concept  
Align CLIP image embeddings with CLAP audio embeddings through a shared text interface.

### Why This Branch Matters  
A strong and robust **semantic understanding baseline**, ideal for object/scene → concept-level mapping.  
Represents the **semantic alignment** category.

### Alignment Focus  
- **Semantic match** ⭐⭐⭐⭐  

---

## 3. `emo-clip/` — **Representative of *Affective Alignment***
### Concept  
Map image emotion (valence/arousal) directly to music emotion features.

### Why This Branch Matters  
This branch specializes in **mood and emotional resonance**, ideal for affect-driven recommendations.  
Represents **affective alignment**.

### Alignment Focus  
- **Affective alignment** ⭐⭐⭐⭐⭐  

---

## 4. `imagebind/` — **Representative of *Cross-Modal Synchrony***
### Concept  
Use a unified embedding space across image, audio, text, and motion.

### Why This Branch Matters  
ImageBind naturally captures **visual motion ↔ audio rhythm**, timbre, and multimodal energy coherence.  
Represents **cross-modal synchrony**.

### Alignment Focus  
- **Cross-modal synchrony** ⭐⭐⭐⭐⭐  

---

# Summary Comparison

| Model | Representative Role | Semantic | Thematic | Energy | Color–Timbre | Rhythm↔Motion | Affective |
|-------|---------------------|----------|----------|--------|--------------|---------------|-----------|
| **Text-Pivot** | Thematic | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐ | ⭐⭐ |
| **CLIP×CLAP** | Semantic | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐ | ⭐ |
| **Emo-CLIP** | Affective | ⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| **ImageBind** | Cross-modal synchrony | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

# Repository Structure
main/
├── clip-x-clap/ # Semantic alignment branch

├── text-pivot/ # Thematic alignment branch

├── emo-clip/ # Affective alignment branch

├── imagebind/ # Cross-modal synchrony branch

└── evaluation/


Each folder contains its own experiments, notebooks, and model implementation.

---

# Research Goal

This project aims to:

1. Benchmark and compare **diverse model combinations**  
2. Understand which alignment strategies best translate **image mood, theme, and emotion** into meaningful music recommendations  
3. Identify the **optimal retrieval architecture** for real-world image-driven music recommendation  
4. Produce clear insights on model behavior across alignment dimensions

The output of this research phase will directly inform both the architecture and ranking pipeline of the final product.

---

# Product Direction (2025)

After selecting the best-performing models and combinations,  
Models developed based on InsightHub will evolve into an **SNS-oriented product** where:

- Users post images  
- The system interprets mood, theme, emotion, and synchrony cues  
- A tailored music recommendation is delivered instantly  
- Recommendations can be shared, remixed, or curated socially

This README reflects the research foundation that will lead into the final user-facing application.

---
