export const rocPoints = [
  { fpr: 0, tpr: 0 },
  { fpr: 0.02, tpr: 0.41 },
  { fpr: 0.05, tpr: 0.68 },
  { fpr: 0.08, tpr: 0.8 },
  { fpr: 0.12, tpr: 0.87 },
  { fpr: 0.18, tpr: 0.91 },
  { fpr: 0.28, tpr: 0.94 },
  { fpr: 0.42, tpr: 0.96 },
  { fpr: 0.6, tpr: 0.98 },
  { fpr: 1, tpr: 1 },
];

export const prPoints = [
  { recall: 0, precision: 1 },
  { recall: 0.12, precision: 0.99 },
  { recall: 0.28, precision: 0.97 },
  { recall: 0.44, precision: 0.95 },
  { recall: 0.58, precision: 0.94 },
  { recall: 0.7, precision: 0.93 },
  { recall: 0.8, precision: 0.91 },
  { recall: 0.88, precision: 0.88 },
  { recall: 0.94, precision: 0.82 },
  { recall: 1, precision: 0.71 },
];

export const lossPoints = Array.from({ length: 25 }, (_, i) => {
  const epoch = i + 1;
  const a = 0.084 * Math.exp(-epoch / 6.2) + 0.00087;
  const b = 0.21 * Math.exp(-epoch / 7.4) + 0.0317;
  return {
    epoch,
    streamA: Number(a.toFixed(5)),
    streamB: Number(b.toFixed(5)),
  };
});

export const accuracyPoints = Array.from({ length: 25 }, (_, i) => {
  const epoch = i + 1;
  const acc = 0.924 - 0.28 * Math.exp(-epoch / 5.6);
  const val = 0.908 - 0.3 * Math.exp(-epoch / 6.4);
  return {
    epoch,
    train: Number((acc * 100).toFixed(2)),
    validation: Number((val * 100).toFixed(2)),
  };
});

export const confusion = {
  labels: ["Healthy", "Irregular", "Frozen"],
  matrix: [
    [88, 3, 1],
    [4, 82, 2],
    [1, 2, 85],
  ],
};
