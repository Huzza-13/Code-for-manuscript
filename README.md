# Global Patterns and Predictors of PFAS Accumulation in Terrestrial Plants

This repository contains the Python scripts used for model development, validation, interpretation, and visualization in the study, *Global Patterns and Predictors of PFAS Accumulation in Terrestrial Plants and Implications for Dietary Exposure*.

## Repository contents

- `scripts/`: analysis and figure-generation scripts.
- `data/`: location for an authorized literature-derived input workbook; the workbook is not distributed here.
- `results/`: destination for locally generated model outputs, tables, and figures.

## Final prediction model

The final XGBoost model predicts log10 PFAS concentration from 13 original predictors: plant tissue group and morphotype; soil pH, soil organic matter, exposure concentration, growth duration, and growth temperature; and six molecular descriptors (ALogP, Chi3v, GATS3c, MinPartialCharge, SpMin8_Bhi, and TPSA).

Carbon-chain length and PFAS class are not model inputs. They are retained only for post hoc chemical interpretation in `03_Molecular_correlation.py` and `17_Molecular_SHAP_vs_chain_and_class.py`.

## Reproduction workflow

Install the required packages in a Python environment:

```bash
pip install -r requirements.txt
```

The principal analysis workflow is as follows:

1. `05_XGBoost_nested_optimization.py`: primary nested Study-PFAS-grouped cross-validation and hyperparameter optimization.
2. `06_Fig3a_OOF_performance.py`: out-of-fold performance plot based on the primary validation output.
3. `07_Final_XGBoost_for_SHAP.py`: refits the final XGBoost model and exports the transformed data, fitted pipeline, SHAP metadata, and tuning results required for interpretation.
4. `08_Fig3b_Overall_SHAP.py`, `09_Fig3d_Molecular_SHAP.py`, and `10_Fig3c_Env_Plant_Exposure_SHAP.py`: global, molecular, and plant/environment/exposure SHAP visualizations.
5. `17_Molecular_SHAP_vs_chain_and_class.py`: compound-level molecular-SHAP visualization by carbon-chain length and PFAS class.

Supporting analyses are provided in `11_FigS_Model_screening.py`, `13_SI_Predictor-domain sensitivity analysis.py`, and `14_SI_Nested_CV_fold_diagnostics.py`. `15_Study_grouped_nested_CV_sensitivity.py` and `16_LOSO_nested_CV.py` provide stricter study-level validation analyses. Scripts `01_model_screening.py`, `02_Feature_block_models.py`, `03_Molecular_correlation.py`, and `04_Reduced_model_comparison.py` document exploratory model-screening, predictor-block, descriptor-screening, and reduced-model comparisons; they are retained for analytical transparency and are not the final prediction workflow.

## Input data and local paths

The scripts expect an Excel workbook containing the columns specified within each script, including `Log PFASs concentration`, `Reference`, and `PFASs Name`. Before use, place an authorized copy of the workbook in `data/` and update the input and output path declarations at the beginning of each relevant script. The current scripts preserve the Windows paths used for the original analysis, rather than assuming a repository-relative location.

The literature-derived workbook is not included because it contains records compiled from published sources. Any public redistribution should be checked against the permissions and terms of the underlying sources.

## Outputs

Generated model objects, cross-validation predictions, tables, and figure files are written to the local output directory configured in each script. These outputs are intentionally not tracked in the repository.

## Citation

Please cite the associated manuscript when using this code. The DOI will be added after publication.

## Licence

No licence has yet been selected. Reuse of this repository requires permission from the authors until a licence is added.
