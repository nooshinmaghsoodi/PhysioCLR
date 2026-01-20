
# PhysioCLR: Physiology-Aware Self-Supervised Learning for ECG

This repository provides the official implementation of **PhysioCLR**, a physiology-aware contrastive learning framework for self-supervised representation learning from electrocardiogram (ECG) signals.

PhysioCLR integrates domain knowledge of cardiac electrophysiology into contrastive learning by:
- Feature-guided positive and negative pair selection,
- Heartbeat-shuffling augmentation,
- Peak-aware reconstruction loss,
- A hybrid contrastive + reconstruction objective.

The method is implemented as a **plugin (user module)** for the `fairseq-signals` framework.

---

## Paper

**Domain Knowledge is Power: Leveraging Physiological Priors for Self-Supervised Representation Learning in Electrocardiography**  
Nooshin Maghsoodi, Sarah Nassar, Paul F. R. Wilson, Minh Nguyen Nhat To, Sophia Mannina, Shamel Addas,  
Stephanie Sibley, David Pichora, David Maslove, Purang Abolmaesumi, Parvin Mousavi  
Submitted to *IEEE Transactions on Biomedical Engineering (TBME)*.

---

## Repository Structure

```

PhysioCLR/
├── physioclr_ext/          # Fairseq-signals user module (models, criterions, tasks)
│   ├── models/            
│   ├── criterions/        
│   ├── modules/           
│   ├── augmentations/     
│   ├── data/              
│   └── tasks/             
│
├── scripts/               
├── configs/               
├── preprocess_ecg.sh      
├── pre-training_ecg.sh    
├── finetuning_ecg.sh      
├── inference_ecg.sh       
└── README.md

````

---

## Installation

### 1. Install fairseq-signals

```bash
git clone https://github.com/Jwoo5/fairseq-signals.git
cd fairseq-signals
pip install -e .
````

### 2. Install PhysioCLR plugin

```bash
git clone https://github.com/nooshinmaghsoodi/PhysioCLR.git
cd PhysioCLR
pip install -e .
```

---

## Using PhysioCLR (No Modification to fairseq-signals)

PhysioCLR is loaded via:

```bash
common.user_dir=/path/to/PhysioCLR
```

This allows fairseq-signals to automatically register:

* `wav2vec2_physioCLR` (model)
* `physioclr` (criterion)
* `ecg_feature` (task)
* `ECGFeatDataset` (dataset)

---



## Citation

```bibtex
@article{maghsoodi2025physioclr,
  title   = {Domain Knowledge is Power: Leveraging Physiological Priors for Self-Supervised Representation Learning in Electrocardiography},
  author  = {Maghsoodi, Nooshin and Nassar, Sarah and Wilson, Paul F R and To, Minh Nguyen Nhat and Mannina, Sophia and Addas, Shamel and Sibley, Stephanie and Pichora, David and Maslove, David and Abolmaesumi, Purang and Mousavi, Parvin},
  journal = {IEEE Transactions on Biomedical Engineering},
  year    = {2025}
}
```

---

## License

This project is released under the **MIT License**.

It is designed as a user-extension for the open-source `fairseq-signals` framework and does not redistribute its source code.


