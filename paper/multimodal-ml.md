# Multimodal Machine Learning

## Course content

Check out our comprehsensive tutorial paper [Foundations and Recent Trends in Multimodal Machine Learning: Principles, Challenges, and Open Questions](https://arxiv.org/abs/2209.03430).

[Tutorials on Multimodal Machine Learning](https://cmu-multicomp-lab.github.io/mmml-tutorial/cvpr2022/) at CVPR 2022 and NAACL 2022, slides and videos [here](https://cmu-multicomp-lab.github.io/mmml-tutorial/schedule/).

New course [11-877 Advanced Topics in Multimodal Machine Learning](https://cmu-multicomp-lab.github.io/adv-mmml-course/spring2022/) Spring 2022 @ CMU. It will primarily be reading and discussion-based. We plan to post discussion probes, relevant papers, and summarized discussion highlights every week on the website.

Public course content and lecture videos from [11-777 Multimodal Machine Learning](https://cmu-multicomp-lab.github.io/mmml-course/fall2020/), Fall 2020 @ CMU.

## Table of Contents

* [Survey Papers](#survey-papers)
* [Core Areas](#core-areas)
  * [Vision](#vision)
  * [Multimodal General Learning](#multimodal-general-learning)
  * [Multimodal Larage Model](#multimodal-larage-Model)
  * [Missing or Imperfect Modalities](#missing-or-imperfect-modalities)
  * [Knowledge Graphs and Knowledge Bases](#knowledge-graphs-and-knowledge-bases)
  * [Generative Learning](#generative-learning)
  * [Human in the Loop Learning](#human-in-the-loop-learning)
* [Applications and Datasets](#applications-and-datasets)
  * [Commonsense Reasoning](#commonsense-reasoning)
  * [Multimodal Reinforcement Learning](#multimodal-reinforcement-learning)
  * [Affect Recognition and Multimodal Language](#affect-recognition-and-multimodal-language)
  * [Human AI Interaction](#Human-AI-Interaction)
* [Workshops](#workshops)
* [Tutorials](#tutorials)
* [Courses](#courses)


# Research Papers

## Survey Papers

[Foundations and Trends in Multimodal Machine Learning: Principles, Challenges, and Open Questions](https://arxiv.org/abs/2209.03430), arxiv 2023

[Multimodal Learning with Transformers: A Survey](https://arxiv.org/abs/2206.06488), TPAMI 2023

[Trends in Integration of Vision and Language Research: A Survey of Tasks, Datasets, and Methods](https://doi.org/10.1613/jair.1.11688), JAIR 2021

[Experience Grounds Language](https://arxiv.org/abs/2004.10151), EMNLP 2020

[A Survey of Reinforcement Learning Informed by Natural Language](https://arxiv.org/abs/1906.03926), IJCAI 2019

[Multimodal Machine Learning: A Survey and Taxonomy](https://arxiv.org/abs/1705.09406), TPAMI 2019

[Multimodal Intelligence: Representation Learning, Information Fusion, and Applications](https://arxiv.org/abs/1911.03977), arXiv 2019

[Deep Multimodal Representation Learning: A Survey](https://ieeexplore.ieee.org/abstract/document/8715409), arXiv 2019

[Guest Editorial: Image and Language Understanding](https://link.springer.com/article/10.1007/s11263-017-0993-y), IJCV 2017

[Representation Learning: A Review and New Perspectives](https://arxiv.org/abs/1206.5538), TPAMI 2013

[A Survey of Socially Interactive Robots](https://www.cs.cmu.edu/~illah/PAPERS/socialroboticssurvey.pdf), 2003

## Core Areas

### Vision
[Segment Anything](https://arxiv.org/pdf/2304.02643, https://segment-anything.com/), 2023

### Multimodal General Learning
[Self-Supervised Learning in Event Sequences: A Comparative Study and Hybrid Approach of Generative Modeling and Contrastive Learning](https://arxiv.org/abs/2401.15935), arXiv 2024

[Identifiability Results for Multimodal Contrastive Learning](https://arxiv.org/abs/2303.09166), ICLR 2023 [[code]](https://github.com/imantdaunhawer/multimodal-contrastive-learning)

[Unpaired Vision-Language Pre-training via Cross-Modal CutMix](https://arxiv.org/abs/2206.08919), ICML 2022.

[Balanced Multimodal Learning via On-the-fly Gradient Modulation](https://arxiv.org/abs/2203.15332), CVPR 2022

[Robust Contrastive Learning against Noisy Views](https://arxiv.org/abs/2201.04309), arXiv 2022

[Cooperative Learning for Multi-view Analysis](https://arxiv.org/abs/2112.12337), arXiv 2022

[Unsupervised Voice-Face Representation Learning by Cross-Modal Prototype Contrast](https://arxiv.org/abs/2204.14057), IJCAI 2021 [[code]](https://github.com/Cocoxili/CMPC)

[Towards a Unified Foundation Model: Jointly Pre-Training Transformers on Unpaired Images and Text](https://arxiv.org/abs/2112.07074), arXiv 2021

[FLAVA: A Foundational Language And Vision Alignment Model](https://arxiv.org/abs/2112.04482), arXiv 2021

[Transformer is All You Need: Multimodal Multitask Learning with a Unified Transformer](https://arxiv.org/abs/2102.10772), arXiv 2021

[What Makes Multi-modal Learning Better than Single (Provably)](https://arxiv.org/abs/2106.04538), NeurIPS 2021

[Efficient Multi-Modal Fusion with Diversity Analysis](https://dl.acm.org/doi/abs/10.1145/3474085.3475188), ACMMM 2021

[Attention Bottlenecks for Multimodal Fusion](https://arxiv.org/abs/2107.00135), NeurIPS 2021

[VMLoc: Variational Fusion For Learning-Based Multimodal Camera Localization](https://arxiv.org/abs/2003.07289), AAAI 2021

[MultiBench: Multiscale Benchmarks for Multimodal Representation Learning](https://arxiv.org/abs/2107.07502), NeurIPS 2021 [[code]](https://github.com/pliang279/MultiBench)

[Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206), ICML 2021 [[code]](https://github.com/deepmind/deepmind-research/tree/master/perceiver)

[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), arXiv 2021 [[blog]]([blog](https://openai.com/blog/clip/)) [[code]](https://github.com/OpenAI/CLIP)

[VinVL: Revisiting Visual Representations in Vision-Language Models](https://arxiv.org/abs/2101.00529), arXiv 2021 [[blog]](https://www.microsoft.com/en-us/research/blog/vinvl-advancing-the-state-of-the-art-for-vision-language-models/?OCID=msr_blog_VinVL_fb) [[code]](https://github.com/pzzhang/VinVL)

[Learning Transferable Visual Models From Natural Language Supervision](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language.pdf), arXiv 2020 [[blog]](https://openai.com/blog/clip/) [[code]](https://github.com/openai/CLIP)

[12-in-1: Multi-Task Vision and Language Representation Learning](https://arxiv.org/abs/1912.02315), CVPR 2020 [[code]](https://github.com/facebookresearch/vilbert-multi-task)

[Multi-Task Learning of Hierarchical Vision-Language Representation](https://arxiv.org/abs/1812.00500), CVPR 2019

[Tensor Fusion Network for Multimodal Sentiment Analysis](https://arxiv.org/abs/1707.07250), EMNLP 2017 [[code]](https://github.com/A2Zadeh/TensorFusionNetwork)

### Multimodal Larage Model

### Missing or Imperfect Modalities

[UniMF: A Unified Multimodal Framework for Multimodal Sentiment Analysis in Missing Modalities and Unaligned Multimodal Sequences](https://ieeexplore.ieee.org/document/10339893), 2024

[A Variational Information Bottleneck Approach to Multi-Omics Data Integration](https://arxiv.org/abs/2102.03014), AISTATS 2021 [[code]](https://github.com/chl8856/DeepIMV)

[SMIL: Multimodal Learning with Severely Missing Modality](https://arxiv.org/abs/2103.05677), AAAI 2021

[Factorized Inference in Deep Markov Models for Incomplete Multimodal Time Series](https://arxiv.org/abs/1905.13570), arXiv 2019

[Learning Representations from Imperfect Time Series Data via Tensor Rank Regularization](https://arxiv.org/abs/1907.01011), ACL 2019

[Multimodal Deep Learning for Robust RGB-D Object Recognition](https://arxiv.org/abs/1507.06821), IROS 2015

### Knowledge Graphs and Knowledge Bases

[MMKG: Multi-Modal Knowledge Graphs](https://arxiv.org/abs/1903.05485), ESWC 2019

[Answering Visual-Relational Queries in Web-Extracted Knowledge Graphs](https://arxiv.org/abs/1709.02314), AKBC 2019

[Embedding Multimodal Relational Data for Knowledge Base Completion](https://arxiv.org/abs/1809.01341), EMNLP 2018

### Generative Learning

[MMVAE+: Enhancing the Generative Quality of Multimodal VAEs without Compromises](https://openreview.net/forum?id=sdQGxouELX), ICLR 2023 [[code]](https://github.com/epalu/mmvaeplus)

[On the Limitations of Multimodal VAEs](https://arxiv.org/abs/2110.04121), ICLR 2022 [[code]](https://openreview.net/attachment?id=w-CPUXXrAj&name=supplementary_material)

[Generalized Multimodal ELBO](https://openreview.net/forum?id=5Y21V0RDBV), ICLR 2021 [[code]](https://github.com/thomassutter/MoPoE)

[Multimodal Generative Learning Utilizing Jensen-Shannon-Divergence](https://arxiv.org/abs/2006.08242), NeurIPS 2020 [[code]](https://github.com/thomassutter/mmjsd)


### Human in the Loop Learning

[Human in the Loop Dialogue Systems](https://sites.google.com/view/hlds-2020/home), NeurIPS 2020 workshop

[Human And Machine in-the-Loop Evaluation and Learning Strategies](https://hamlets-workshop.github.io/), NeurIPS 2020 workshop

[Human-centric dialog training via offline reinforcement learning](https://arxiv.org/abs/2010.05848), EMNLP 2020 [[code]](https://github.com/natashamjaques/neural_chat/tree/master/BatchRL)


## Applications and Datasets

### Language and Visual QA

[TAG: Boosting Text-VQA via Text-aware Visual Question-answer Generation](https://arxiv.org/abs/2208.01813), arXiv 2022 [[code]](https://github.com/HenryJunW/TAG)

[Learning to Answer Questions in Dynamic Audio-Visual Scenarios](https://arxiv.org/abs/2203.14072), CVPR 2022

[SUTD-TrafficQA: A Question Answering Benchmark and an Efficient Network for Video Reasoning over Traffic Events](https://openaccess.thecvf.com/content/CVPR2021/html/Xu_SUTD-TrafficQA_A_Question_Answering_Benchmark_and_an_Efficient_Network_for_CVPR_2021_paper.html), CVPR 2021 [[code]](https://github.com/SUTDCV/SUTD-TrafficQA)

[MultiModalQA: complex question answering over text, tables and images](https://openreview.net/forum?id=ee6W5UgQLa), ICLR 2021

[ManyModalQA: Modality Disambiguation and QA over Diverse Inputs](https://arxiv.org/abs/2001.08034), AAAI 2020 [[code]](https://github.com/hannandarryl/ManyModalQA)

[Iterative Answer Prediction with Pointer-Augmented Multimodal Transformers for TextVQA](https://arxiv.org/abs/1911.06258), CVPR 2020

### Commonsense Reasoning

[Adventures in Flatland: Perceiving Social Interactions Under Physical Dynamics](https://www.tshu.io/HeiderSimmel/CogSci20/Flatland_CogSci20.pdf), CogSci 2020

[A Logical Model for Supporting Social Commonsense Knowledge Acquisition](https://arxiv.org/abs/1912.11599), arXiv 2019

[Heterogeneous Graph Learning for Visual Commonsense Reasoning](https://arxiv.org/abs/1910.11475), NeurIPS 2019

### Multimodal Reinforcement Learning

[MiniHack the Planet: A Sandbox for Open-Ended Reinforcement Learning Research](https://arxiv.org/abs/2109.13202), NeurIPS 2021 [[code]](https://github.com/facebookresearch/minihack)

[Imitating Interactive Intelligence](https://arxiv.org/abs/2012.05672), arXiv 2020

[Grounded Language Learning Fast and Slow](https://arxiv.org/abs/2009.01719), ICLR 2021

[RTFM: Generalising to Novel Environment Dynamics via Reading](https://arxiv.org/abs/1910.08210), ICLR 2020 [[code]](https://github.com/facebookresearch/RTFM)

[Embodied Multimodal Multitask Learning](https://arxiv.org/abs/1902.01385), IJCAI 2020

[Learning to Speak and Act in a Fantasy Text Adventure Game](https://arxiv.org/abs/1903.03094), arXiv 2019 [[code]](https://parl.ai/projects/light/)

[Language as an Abstraction for Hierarchical Deep Reinforcement Learning](https://arxiv.org/abs/1906.07343), NeurIPS 2019


### Affect Recognition and Multimodal Language

[Deep-HOSeq: Deep Higher-Order Sequence Fusion for Multimodal Sentiment Analysis](https://arxiv.org/pdf/2010.08218.pdf), ICDM 2020 

[End-to-end Facial and Physiological Model for Affective Computing and Applications](https://arxiv.org/abs/1912.04711), arXiv 2019

[Affective Computing for Large-Scale Heterogeneous Multimedia Data: A Survey](https://arxiv.org/abs/1911.05609), ACM TOMM 2019

[Towards Multimodal Sarcasm Detection (An Obviously_Perfect Paper)](https://arxiv.org/abs/1906.01815), ACL 2019 [[code]](https://github.com/soujanyaporia/MUStARD)

[Multi-modal Approach for Affective Computing](https://arxiv.org/abs/1804.09452), EMBC 2018

[Multimodal Language Analysis with Recurrent Multistage Fusion](https://arxiv.org/abs/1808.03920), EMNLP 2018

[Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph](http://aclweb.org/anthology/P18-1208), ACL 2018 [[code]](https://github.com/A2Zadeh/CMU-MultimodalSDK)

[Multi-attention Recurrent Network for Human Communication Comprehension](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/17390/16123), AAAI 2018 [[code]](https://github.com/A2Zadeh/CMU-MultimodalSDK)

[End-to-End Multimodal Emotion Recognition using Deep Neural Networks](https://arxiv.org/abs/1704.08619), arXiv 2017


### Human AI Interaction

[Multimodal Human Computer Interaction: A Survey](https://link.springer.com/chapter/10.1007/11573425_1), HCI 2005

[Affective multimodal human-computer interaction](https://dl.acm.org/doi/10.1145/1101149.1101299), Multimedia 2005


# Courses

[CMU 11-777 Multimodal Machine Learning](https://cmu-multicomp-lab.github.io/mmml-course/fall2022/)

[CMU 11-877 Advanced Topics in Multimodal Machine Learning](https://cmu-multicomp-lab.github.io/adv-mmml-course/spring2023/)

[CMU 05-618, Human-AI Interaction](https://haiicmu.github.io/)

[CMU 11-777, Advanced Multimodal Machine Learning](https://piazza.com/cmu/fall2018/11777/resources)

[Stanford CS422: Interactive and Embodied Learning](http://cs422interactive.stanford.edu/)

[CMU 16-785, Integrated Intelligence in Robotics: Vision, Language, and Planning](http://www.cs.cmu.edu/~jeanoh/16-785/)

[CMU 10-808, Language Grounding to Vision and Control](https://katefvision.github.io/LanguageGrounding/)

[CMU 11-775, Large-Scale Multimedia Analysis](https://sites.google.com/a/is.cs.cmu.edu/lti-speech-classes/11-775-large-scale-multimedia-analysis)

[MIT 6.882, Embodied Intelligence](https://phillipi.github.io/6.882/)

[Georgia Tech CS 8803, Vision and Language](http://www.prism.gatech.edu/~arjun9/CS8803_CVL_Fall17/)

[Virginia Tech CS 6501-004, Vision & Language](http://www.cs.virginia.edu/~vicente/vislang/)

[Machine Learning Career: A Comprehensive Guide](https://www.scaler.com/blog/machine-learning-career/)
