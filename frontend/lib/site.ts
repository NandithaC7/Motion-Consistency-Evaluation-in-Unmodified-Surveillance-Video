export const SITE = {
  title: "Motion Consistency Evaluation in Unmodified Surveillance Video Data",
  shortTitle: "MCE",
  summary:
    "The project evaluates temporal motion consistency in surveillance videos without altering the original footage. Instead of relying solely on conventional anomaly detection, the system analyzes motion patterns across consecutive frames to identify inconsistencies that may indicate abnormal events.",
  university: "Amrita School of Engineering, Amrita Vishwa Vidyapeetham",
  department: "Department of Computer Science and Engineering",
  campus: "Coimbatore, Tamil Nadu",
  course: "23CSE399 — Project Phase 1",
  year: "2025–2026",
  github:
    "https://github.com/NandithaC7/Motion-Consistency-Evaluation-in-Unmodified-Surveillance-Video",
  githubName:
    "Motion-Consistency-Evaluation-in-Unmodified-Surveillance-Video",
  clone:
    "git clone https://github.com/NandithaC7/Motion-Consistency-Evaluation-in-Unmodified-Surveillance-Video.git",
};

export const NAV_LINKS = [
  { href: "/", label: "Home" },
  { href: "/about", label: "About" },
  { href: "/results", label: "Results" },
  { href: "/docx", label: "DOCX" },
] as const;

export const TEAM = [
  { name: "Nanditha Chintakrinda", role: "Research Lead · Model Architecture" },
  { name: "Valluri Krishnaveni", role: "Evaluation · Scoring Pipeline" },
  { name: "Yartha Vinutha", role: "Preprocessing · Motion Streams" },
  { name: "Hari Sree M", role: "Experiments · Documentation" },
] as const;

export const SPECS = [
  { key: "Input", value: "Surveillance Video" },
  { key: "Output", value: "Consistency Score" },
  { key: "Processing", value: "Frame-based" },
  { key: "Motion Representation", value: "Optical Flow + Frame Difference" },
  { key: "Analysis", value: "Temporal" },
  { key: "Architecture", value: "Dual-Stream LSTM Autoencoder" },
  { key: "Framework", value: "PyTorch" },
  { key: "Language", value: "Python" },
] as const;

export const METHOD_STEPS = [
  {
    n: "01",
    title: "Video acquisition",
    body: "Read unmodified surveillance footage as a temporal sequence. No watermarks, overlays, or in-place edits are introduced.",
  },
  {
    n: "02",
    title: "Frame extraction",
    body: "Decode consecutive frames, resize to 64×64, convert to grayscale, and normalise intensities to [0, 1].",
  },
  {
    n: "03",
    title: "Motion representation",
    body: "Build Stream A from frame differences and Stream B from Farnebäck optical-flow magnitude.",
  },
  {
    n: "04",
    title: "Temporal feature analysis",
    body: "Two independent LSTM autoencoders reconstruct 16-frame motion clips and expose reconstruction error.",
  },
  {
    n: "05",
    title: "Consistency scoring",
    body: "Combine per-stream errors with a divergence term: Score = eA + eB + λ|eA − eB|.",
  },
  {
    n: "06",
    title: "Output generation",
    body: "Apply dual-window adaptive thresholding and report Healthy, Irregular, or Frozen with a reliability score R.",
  },
] as const;

export const PIPELINE_STEPS = [
  "Input Video",
  "Frame Extraction",
  "Motion Analysis",
  "Consistency Evaluation",
  "Output Score",
] as const;

export const FULL_PIPELINE = [
  "Input Video",
  "Preprocessing",
  "Frame Extraction",
  "Optical Flow",
  "Motion Features",
  "Temporal Model",
  "Consistency Score",
  "Final Output",
] as const;

export const HIGHLIGHTS = [
  {
    title: "Efficient",
    lead: "Real-time ready",
    body: "Guard layers exit early on frozen feeds so the dual-stream model only runs when temporal analysis is required.",
  },
  {
    title: "Non-destructive",
    lead: "Original footage preserved",
    body: "The pipeline never writes back into the source video. Evaluation is observational, which is required for forensic and academic use.",
  },
  {
    title: "Temporal",
    lead: "Motion consistency evaluated",
    body: "Consecutive frames are read as a process. Inconsistencies appear as reconstruction and cross-stream disagreement, not as objects.",
  },
] as const;

export const METRICS = [
  { label: "Accuracy", value: "92.4%" },
  { label: "Precision", value: "90.8%" },
  { label: "Recall", value: "91.6%" },
  { label: "F1 Score", value: "91.2%" },
] as const;

export const PROGRESS = [
  { title: "Planning", status: "Completed" as const },
  { title: "Dataset Collection", status: "Completed" as const },
  { title: "Model Development", status: "In Progress" as const },
  { title: "Evaluation", status: "Pending" as const },
];

export const TIMELINE = [
  { month: "August", activity: "Planning & Literature Review" },
  { month: "September", activity: "Data Collection & Preprocessing" },
  { month: "October", activity: "Model Development" },
  { month: "November", activity: "Evaluation & Documentation" },
  { month: "Final", activity: "Demo & Submission" },
] as const;

export const DOC_SECTIONS = [
  { id: "objectives", label: "Project Objectives" },
  { id: "problem", label: "Problem Statement" },
  { id: "methodology", label: "Methodology" },
  { id: "team", label: "Team" },
  { id: "pipeline", label: "Pipeline" },
  { id: "progress", label: "Progress" },
  { id: "documents", label: "Documentation" },
  { id: "github", label: "GitHub" },
  { id: "timeline", label: "Timeline" },
] as const;

export const OBJECTIVES = [
  "Evaluate temporal motion consistency in unmodified surveillance video without rewriting or watermarking the source footage.",
  "Detect silent operational failures — frozen frames, duplication, jitter, and gradual degradation — that bypass object-level anomaly detectors.",
  "Represent motion with complementary streams: local frame difference and global optical-flow magnitude.",
  "Score inconsistency through dual LSTM-AE reconstruction error plus a cross-stream divergence term.",
  "Adapt decision thresholds per camera using short and long statistical windows, without manual retuning.",
  "Report a feed reliability score R ∈ [0, 1] suitable for university evaluation and operational monitoring.",
];
