# Stage 5E — reliable demonstration audit for the existing scratch model

**24 November 2026. Inference only.** This audit used the unchanged Stage 4A YOLO11n scratch checkpoint `runs/scratch_yolo11n_stage3c_experimental_b16/weights/best.pt` (SHA-256 `1bb3228630eb609b71cfb773ba67212c7641e0d477bdc4b79883ecd747c28a24`). It ran on **48 images from the existing Stage 3C validation split**: 12 each selected for a provisional pistol, backpack, safety-vest or helmet label. Selection used fixed seeds and distinct provisional component groups within each class. Inference used image size 640, confidence **0.25**, NMS IoU **0.70**, and the checkpoint's fixed six-class map. The [audit CSV](../reports/stage5e/audit.csv) records image/source group, prediction classes and scores, maximum same-class IoU against provisional boxes, and largest predicted area fraction. [Provenance](../reports/stage5e/provenance.json) records the settings. This is a small demonstration audit, **not independent accuracy** or a replacement for Stage 4B metrics. Provisional labels are not human verified; IoU only helps triage visual examples.

## Visual findings

| Focus class | Sample observations at confidence 0.25 | Visual conclusion |
|---|---|---|
| Backpack | 11/12 images have a same-class box overlapping a provisional box at IoU ≥ 0.5; 10/12 at ≥ 0.8. | **Strongest demonstration class**, especially isolated product photos. Boxes generally follow the bag outline. One scene had no backpack prediction; two product images also acquired implausible helmet labels. |
| Safety vest | 12/12 have at least one overlapping vest box at IoU ≥ 0.5; 10/12 at ≥ 0.8. | Good secondary demonstration on carefully chosen images of people wearing vests. One high-overlap match does not mean every vest/person in a crowded image was found; some images show extra or imperfect boxes. |
| Helmet | 7/12 have an overlapping helmet box at IoU ≥ 0.5; five have no helmet prediction. | Mixed. Several hard-hat images localize well, but misses and inappropriate source taxonomy make it less dependable for a live demo. |
| Pistol | 6/12 have an overlapping pistol box at IoU ≥ 0.5; five have no pistol prediction. | **Do not demonstrate as a reliable robbery/CCTV detector.** Some isolated gun product photos localize plausibly, but misses and oversized boxes make results highly context dependent. |

The [backpack contact sheet](../reports/stage5e/backpack_contact_sheet.jpg), [safety-vest contact sheet](../reports/stage5e/safety_vest_contact_sheet.jpg), [helmet contact sheet](../reports/stage5e/helmet_contact_sheet.jpg) and [pistol contact sheet](../reports/stage5e/pistol_contact_sheet.jpg) show provisional annotations on the left and actual model predictions on the right. Every individual comparison is under [examples](../reports/stage5e/examples). The numeric counts above describe this selected sample only and may overstate performance on natural video or other sources.

## Visually suitable examples

For a **short image-sequence demonstration**, use separate still images with a clear caption that these are selected validation examples, not consecutive video frames or independent CCTV accuracy. Strong backpack candidates with localized predicted boxes are:

1. [Black Safari backpack](../reports/stage5e/examples/backpack_41485109fdf1919d0c12.jpg), same-class IoU 0.979.
2. [Black and grey HRX backpack](../reports/stage5e/examples/backpack_8c1075078f66e20bc039.jpg), IoU 0.941.
3. [Blue HRX backpack](../reports/stage5e/examples/backpack_ebcb35f09a0a5e92b978.jpg), IoU 0.936.
4. [Blue Safari backpack](../reports/stage5e/examples/backpack_f7c8b829136f7e776ef8.jpg), IoU 0.879.

These are visually plausible isolated bag detections, but are similar product-style images. A more varied, secondary vest sequence can use [two workers in green vests](../reports/stage5e/examples/safety_vest_3a81a64097fac7ad07eb.jpg), [one worker in an orange vest](../reports/stage5e/examples/safety_vest_44c2df5ab8b9c056c0df.jpg), and [two workers in orange/green vests](../reports/stage5e/examples/safety_vest_e55de220eea28a6db68c.jpg). Show the source image alongside its prediction, with the 0.25 confidence setting visible in the narration or caption. None of these examples has been independently reviewed as ground truth.

## Failure examples to show separately

- [Large false pistol on a construction/helmet scene](../reports/stage5e/examples/helmet_2c39792c8c30bae7fede.jpg): the model predicts a pistol covering **85.4%** of the image, with no corresponding provisional pistol annotation and no visually plausible gun. This is the clearest sampled large false positive.
- [Oversized pistol on a muzzle close-up](../reports/stage5e/examples/pistol_5bded06e4efd4952c859.jpg): predicted box covers **87.4%** of the image while matching its provisional pistol box poorly (IoU 0.288). A gun is visible, but localization is unsuitable for a convincing demo.
- [Two pistols in a game image, neither detected](../reports/stage5e/examples/pistol_c1fede89228329d9ed77.jpg) and [person holding a pistol, no prediction](../reports/stage5e/examples/pistol_82c3538effd7e13bf8bd.jpg): examples of missed detections at 0.25. The former also illustrates that source domains include game imagery.
- [Backpack scene with no backpack prediction](../reports/stage5e/examples/backpack_0fb11699932d55003be1.jpg) and [vest group with extra/uneven boxes](../reports/stage5e/examples/safety_vest_ab82547de3a9ae48d621.jpg) limit the otherwise stronger classes.

## Recommendation

Demonstrate **backpack detection** as the primary honest capability of this checkpoint, using the four selected still images as an image sequence. Safety vest is an optional second sequence using the three examples above. State that the images were selected from a provisional validation split and that results do not measure independent CCTV performance. Do not imply continuous video tracking or reliable pistol detection. If the mentor expects a robbery clip, present it as a known failure case rather than a success. No training, dataset/split change, download, full test rerun, confusion-matrix update, or checkpoint change was performed.
