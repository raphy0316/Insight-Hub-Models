# InsightHub – Image→Music Multimodal Retrieval  
### *Research Branch Overview (CLIP×CLAP · Text-Pivot · Emo-CLIP · ImageBind)*

This repository hosts multiple research branches exploring different approaches for recommending music from user-uploaded images.  
Each branch implements a distinct **multimodal alignment strategy**, and all models are compared through the InsightHub platform.

---

## 🔑 Cross-Modal Alignment Dimensions

We evaluate each approach based on the following core dimensions:

- **Cross-modal synchrony**
  - Energy coherence  
  - Temporal rhythm ↔ visual motion  
  - Color–timbre association  
- **Semantic match**
- **Thematic match**
- **Affective (emotional) alignment**

Different models specialize in different subsets of these factors.

---

# 🌿 Branch Overview

Below is a high-level summary of each research branch.  
(Full pipelines and implementation details are documented inside each branch directory.)

---

## 1. `text-pivot/`
### **Concept:** Normalize both images and music into the **text domain**, then match based on shared concepts/themes.

The Text-Pivot approach extracts keywords or descriptive text from both the image and the music, embeds them in the same text space, and computes similarity.  
It achieves the strongest **semantic and thematic** alignment.

**Strengths**
- Excellent semantic and thematic reasoning  
- Human-interpretable, concept-level matching  
- Utilizes metadata (captions, hashtags, lyrics, editorial tags)

**Alignment Focus**
| Dimension | Contribution |
|----------|--------------|
| Semantic match | ⭐⭐⭐⭐⭐ |
| Thematic match | ⭐⭐⭐⭐ |
| Energy coherence | ⭐⭐⭐ |
| Affective alignment | ⭐⭐ |
| Color–timbre association | ⭐ |

---

## 2. `clip-x-clap/`
### **Concept:** Match CLIP image embeddings with CLAP audio embeddings through their shared text interface.

A robust baseline that performs direct multimodal alignment using widely-adopted embedding models.

**Strengths**
- Fast, stable, and zero-shot friendly  
- Good general semantic alignment  
- Lightweight compared to ImageBind

**Alignment Focus**
| Dimension | Contribution |
|----------|--------------|
| Semantic match | ⭐⭐⭐⭐ |
| Thematic match | ⭐⭐⭐ |
| Energy coherence | ⭐⭐ |
| Affective alignment | ⭐ |
| Color–timbre association | ⭐⭐ |

---

## 3. `emo-clip/`
### **Concept:** Map image emotion (valence/arousal) to the emotional spectrum of music.

This branch focuses on **mood-driven** recommendations rather than semantics or themes.

**Strengths**
- Best emotional alignment across all models  
- Works for portraits, landscapes, and abstract scenes  
- Direct mapping between image mood and audio emotion features

**Alignment Focus**
| Dimension | Contribution |
|----------|--------------|
| Affective alignment | ⭐⭐⭐⭐⭐ |
| Energy coherence | ⭐⭐⭐⭐ |
| Color–timbre association | ⭐⭐⭐ |
| Semantic/Thematic match | Low |

---

## 4. `imagebind/`
### **Concept:** Use a shared embedding space covering image, audio, text, and motion.

ImageBind provides the most comprehensive cross-modal synchrony, capturing rhythm, intensity, texture, and atmosphere.

**Strengths**
- Strongest overall multimodal alignment  
- Natural mapping between visual motion ↔ audio rhythm  
- High upper-bound performance for complex scenes

**Alignment Focus**
| Dimension | Contribution |
|----------|--------------|
| Cross-modal synchrony | ⭐⭐⭐⭐⭐ |
| Rhythm ↔ visual motion | ⭐⭐⭐⭐⭐ |
| Color–timbre association | ⭐⭐⭐⭐ |
| Energy coherence | ⭐⭐⭐⭐ |
| Semantic/Thematic match | ⭐⭐⭐⭐ |

---

# 📊 Summary Comparison

| Model | Semantic | Thematic | Energy | Color–Timbre | Rhythm↔Motion | Affective |
|-------|----------|----------|--------|--------------|---------------|-----------|
| **Text-Pivot** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐ | ⭐⭐ |
| **CLIP×CLAP** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐ | ⭐ |
| **Emo-CLIP** | ⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| **ImageBind** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

# 📂 Repository Structure
main/

├── clip-x-clap/

├── text-pivot/

├── emo-clip/

├── imagebind/

└── evaluation/



Each folder represents an independent research track with its own methodology, experiments, and documentation.

---

# 🚀 Roadmap

Our goal for 2025 is to build a production-ready **image-driven music recommendation service**, supported by InsightHub’s evaluation framework for model comparison, A/B testing, and UX prototyping.

---

