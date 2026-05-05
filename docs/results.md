# Results

## Headline numbers (model_v002, 5-class taxonomy)

- **Validation accuracy**: 96.6% on 59 held-out samples
- **Per-class accuracy**:
  - concrete-plaster: 100% (9/9)
  - fabric: 100% (21/21)
  - wood: 100% (9/9)
  - nature: 91.7% (11/12)
  - brick: 87.5% (7/8)

## Confusion matrix

The model made 2 errors out of 59 validation samples. Both errors fall into the `fabric` predicted column — but for entirely different visual reasons:

- **One brick predicted as fabric.** Painted-white brick with smooth surface and minimal mortar contrast. The brick class concept extends to painted variants but the model has too few painted examples to learn the subcategory reliably.
- **One nature predicted as fabric.** A close-up texture with fine repeating structure that reads as woven fabric to the model. Worth opening and deciding whether it belongs in `nature` or represents a class boundary issue.

## Honest evaluation

The validation accuracy is computed on the same held-out indices that the model was scored against during training, with a seed cross-check between `val_indices.json` and `model_metadata.json` — the script refuses to evaluate if those don't match. The 96.6% number is reproducible and reflects performance on samples the model has never seen.

## Confidence-thresholded behavior on out-of-distribution samples

Tested against textures outside the 5 trained classes:

| Test image | Top prediction | Confidence | Routing |
|------------|---------------|------------|---------|
| metal_test_01 | fabric | 73.2% | REVIEW |
| metal_test_02 | nature | 40.2% | REVIEW |
| water_test_01 | fabric | 80.9% | REVIEW |

The model correctly distributes uncertainty across multiple classes when shown content outside its training distribution, and the 85% confidence threshold correctly routes all three to the Review folder rather than auto-sorting them. This is the human-in-the-loop design working as intended.

## Limitations

- Confidence intervals on 59 val samples are wide. A point estimate of 96.6% is consistent with a true accuracy anywhere from ~88% to ~99%. Don't read narrow accuracy comparisons (96% vs 97%) as meaningful at this dataset size.
- Class balance matters. Fabric had 21 val samples, brick had 8. The model's per-class numbers are correspondingly more reliable for fabric than brick.
- The validated taxonomy has clean visual separation between classes. Performance on taxonomies with fine-grained or visually overlapping classes (e.g. distinguishing wood species, or polished vs matte concrete) is unknown without retraining.
