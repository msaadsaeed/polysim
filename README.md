# POLYSIM 2026 Grand Challenge  
## Polyglot Speaker Identification with Missing Modality

<!--## $\textcolor{green}{Accepting \space Submissions!}$ -->

[Challenge Webpage](https://mmosc.github.io/fame2027.github.io/index.html#home)  
[Evaluation Plan](https://arxiv.org/abs/2603.24569)

---

# Baseline

Baseline systems are provided to benchmark multimodal speaker classification under missing-modality and cross-lingual conditions.

**Baseline**
- [FOP](https://arxiv.org/abs/2112.10483)

### Baseline Results

| Configuration       | Phase    | P3 (Face-Audio Eng.) | P4 (Audio Eng.) | P5 (Face-Audio Urdu) | P6 (Audio Urdu) | Avg.  |
|---------------------|----------|----------------------|-----------------|----------------------|-----------------|-------|
| Face-Audio (Eng.)   | Progress | 97.44                | 37.75           | 98.48                | 31.70           | 66.34 |
| Face-Audio (Eng.)   | Eval     | 98.82                | 52.53           | 98.27                | 43.87           | 73.37 |

---

## Task

The POLYSIM 2026 challenge addresses **closed-set speaker classification** using **audio (voice)** and **visual (face)** modalities.

The goal is to classify a speaker’s identity from a given sample under the following real-world challenges:

- One modality (face) may be **completely missing** at test time  
- The **test language may differ** from the training language  
- A **single unified model** must be used across all conditions  

Participants are required to design models that are **robust, modality-agnostic, and cross-lingual**.

---

## Task Settings

The challenge consists of **four task settings**, covering multimodal, missing-modality, and cross-lingual scenarios.

### P3: In-Language Multimodal
- **Training**: Audio + Face  
- **Testing**: Audio + Face  
- **Language**: Same  

Standard multimodal speaker classification setting.

---

### P4: Missing-Modality (Audio-Only)
- **Training**: Audio + Face  
- **Testing**: Audio only  
- **Language**: Same  

The face modality is completely missing at test time.  
**No retraining is allowed.**

---

### P5: Cross-Lingual Multimodal
- **Training**: Audio + Face  
- **Testing**: Audio + Face  
- **Language**: Different  

Evaluates cross-lingual generalization with full modality availability.

---

### P6: Cross-Lingual Missing-Modality
- **Training**: Audio + Face  
- **Testing**: Audio only  
- **Language**: Different  

The most challenging setting, combining:
- Cross-lingual testing  
- Missing face modality at inference  

---

### Task Settings Summary

<table border="1" align="center">
  <tr>
    <th>Setting</th>
    <th>Training Modalities</th>
    <th>Testing Modalities</th>
    <th>Language</th>
  </tr>
  <tr>
    <td align="center">P3</td>
    <td align="center">Audio + Face</td>
    <td align="center">Audio + Face</td>
    <td align="center">Same</td>
  </tr>
  <tr>
    <td align="center">P4</td>
    <td align="center">Audio + Face</td>
    <td align="center">Audio only</td>
    <td align="center">Same</td>
  </tr>
  <tr>
    <td align="center">P5</td>
    <td align="center">Audio + Face</td>
    <td align="center">Audio + Face</td>
    <td align="center">Cross-lingual</td>
  </tr>
  <tr>
    <td align="center">P6</td>
    <td align="center">Audio + Face</td>
    <td align="center">Audio only</td>
    <td align="center">Cross-lingual</td>
  </tr>
</table>

---

## Dataset

### Overview

The dataset MavCeleb consists of **paired speech audio and face images/video frames** collected from multiple speakers across **multiple languages**.

### Modalities
- **Audio**: Speech segments  
- **Visual**: Face images or face tracks  
- **Labels**: Speaker ID  

### Data Splits
- Training set - [Raw data](https://drive.google.com/file/d/1bRFbL9jOcmauEMJlEN_-4DvbpL2SFwlz/view?usp=sharing), [features](https://drive.google.com/file/d/11gjdlJgRxWcoNU9h-xlD8W0171Q6nvAA/view?usp=sharing), [CSV file](https://drive.google.com/file/d/1V_Wctz02J1wA47rTAZ3uIXiCkw9rYDQd/view?usp=sharing) 
- Dev set  (labels hidden) - [Raw data](https://drive.google.com/file/d/1JRkgFKUo0hVdnPaT8JsrSn06C1hl8g89/view?usp=sharing), [features](https://drive.google.com/file/d/19smDBY2fcvR-vXML2UjGrWc0XyfDO3-l/view?usp=sharing), [CSV file](https://drive.google.com/file/d/1V_Wctz02J1wA47rTAZ3uIXiCkw9rYDQd/view?usp=sharing)
- Test set (labels hidden) - ![Coming Soon](https://img.shields.io/badge/status-coming%20soon-red)

### Missing Modality Setup
- Missing modality occurs **only at test time**
- Missing modality is **explicit and complete** (face absent)
- Training data **always includes both modalities**

---

## Evaluation Protocol

The goal of evaluation is to study:

- Multimodal speaker classification performance  
- Robustness to **missing face modality**  
- Generalization across **unseen languages**  

### Metrics
- **Accuracy**

### Ranking
- Metrics are computed **separately** for P3, P4, P5, and P6  
- **Final ranking** is based on the **average score** across all settings

---

## Submission

Participants must submit a ZIP archive containing CSV files, one per language pair. To create the archive, `zip submission.zip *.csv` from within the directory containing the submission files (do not zip the folder itself). Files must be named as follows:
```
- submission_v1_<phase>_English_English.csv
- submission_v1_<phase>_English_Urdu.csv
```


Where `<phase>` is val (dev) or test (eval). 

## Reference 

```BibTeX
@inproceedings{saeed2022fusion,
  title     = {Fusion and Orthogonal Projection for Improved Face-Voice Association},
  author    = {Saeed, Muhammad Saad and
               Khan, Muhammad Haris and
               Nawaz, Shah and
               Yousaf, Muhammad Haroon and
               Del Bue, Alessio},
  booktitle = {ICASSP 2022 -- IEEE International Conference on Acoustics, Speech and Signal Processing},
  pages     = {7057--7061},
  year      = {2022},
  organization = {IEEE}
}

@inproceedings{nawaz2021cross,
  title     = {Cross-Modal Speaker Verification and Recognition: A Multilingual Perspective},
  author    = {Nawaz, Shah and
               Saeed, Muhammad Saad and
               Morerio, Pietro and
               Mahmood, Arif and
               Gallo, Ignazio and
               Yousaf, Muhammad Haroon and
               Del Bue, Alessio},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {1682--1691},
  year      = {2021}
}
```

