import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 SVAGFormer Training on Kaggle\n",
                "This notebook automatically sets up the environment, clones the SVAGFormer repository, and starts the training pipeline on Kaggle's dual GPU (T4x2) or single P100 environment.\n",
                "\n",
                "**Prerequisites:**\n",
                "1. Make sure **Internet** is enabled in the notebook settings (right panel).\n",
                "2. Make sure your **Accelerator** is set to **GPU T4 x2** or **GPU P100**.\n",
                "3. Attach the MOT17/MOT20/OVIS datasets to this notebook via the 'Add Data' button in Kaggle."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%time\n",
                "# 1. Clone the SVAGFormer repository with submodules\n",
                "!git clone --recurse-submodules https://github.com/wassima-azzouzi/svagformer.git\n",
                "%cd svagformer"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%time\n",
                "# 2. Install dependencies\n",
                "# Note: We might need to handle specific versions for Kaggle's environment\n",
                "!pip install -r requirements.txt\n",
                "!pip install trackeval  # Required by SVAGEval"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 🔗 Link Kaggle Datasets to the Code structure\n",
                "Assuming you attached the datasets into `/kaggle/input/`, we link them to our `data/` folder."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "\n",
                "# Create SVAG-Bench folder structure\n",
                "!python scripts/prepare_datasets.py\n",
                "\n",
                "# Link Kaggle input directories to SVAGFormer data dirs\n",
                "# Example for MOT17 (Change 'mot17-dataset' depending on your Kaggle dataset name)\n",
                "KAGGLE_MOT17 = '/kaggle/input/mot17-dataset'\n",
                "if os.path.exists(KAGGLE_MOT17):\n",
                "    !ln -s {KAGGLE_MOT17}/images data/mot17/images\n",
                "    !ln -s {KAGGLE_MOT17}/annotations data/mot17/annotations\n",
                "    print(\"Linked MOT17 dataset!\")\n",
                "else:\n",
                "    print(\"Please attach the dataset to the Notebook in Kaggle.\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%time\n",
                "# 3. Format Data for FlashVTG & TempRMOT\n",
                "# This converts annotations to QVHighlights format and RMOT format\n",
                "!python utils/dataset.py"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 🏋️ Train Models\n",
                "We train spatial (TempRMOT) and temporal (FlashVTG) independently as specified in the paper."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Train Temporal module on MOT17\n",
                "!python scripts/train_temporal.py --dataset mot17"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Train Spatial module on MOT17\n",
                "# Note: TempRMOT uses PyTorch Distributed Data Parallel\n",
                "!python -m torch.distributed.launch --nproc_per_node=2 scripts/train_spatial.py --dataset mot17"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 📈 Evaluation\n",
                "Generate predictions and score them with SVAGEval to get m-HIoU."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python scripts/evaluate.py --predictions-dir outputs/predictions --gt-dir data/ground_truth --datasets mot17"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("svagformer_kaggle.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated: svagformer_kaggle.ipynb")
