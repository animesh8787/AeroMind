// Builds docs/preread/AeroMind-PreRead.docx. Every number is copied from docs/evidence-report.md.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, Header, Footer,
  AlignmentType, LevelFormat, BorderStyle, WidthType, ShadingType, PageNumber, TabStopType,
} = require("docx");

const ROOT = path.resolve(__dirname, "../..");
const FIG = path.join(ROOT, "docs/preread/figures");
const NAVY = "0B2545", TEAL = "13A89E", GREY = "5B6770", LIGHT = "EAF2F7", AMBER = "E09F3E";
const FONT = "Calibri";
const W = 10466; // A4 width 11906 - 2 x 720 margins

function png(file, widthPx) {
  const b = fs.readFileSync(path.join(FIG, file));
  const w = b.readUInt32BE(16), h = b.readUInt32BE(20);
  return new ImageRun({ type: "png", data: b, transformation: { width: widthPx, height: Math.round((widthPx * h) / w) } });
}
const t = (text, o = {}) => new TextRun({ text, font: FONT, size: 20, ...o });
const p = (children, o = {}) =>
  new Paragraph({ spacing: { after: 100, line: 264 }, ...o, children: Array.isArray(children) ? children : [t(children)] });
const h1 = (text) =>
  new Paragraph({
    spacing: { before: 200, after: 90 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: TEAL, space: 2 } },
    keepNext: true,
    children: [new TextRun({ text, font: FONT, size: 26, bold: true, color: NAVY })],
  });
const bullet = (runs) =>
  new Paragraph({ numbering: { reference: "b", level: 0 }, spacing: { after: 50, line: 258 }, children: Array.isArray(runs) ? runs : [t(runs)] });
const bold = (s) => t(s, { bold: true, color: NAVY });
const note = (s) => p([t(s, { italics: true, size: 17, color: GREY })], { spacing: { before: 60, after: 80 } });

const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const thin = { style: BorderStyle.SINGLE, size: 4, color: "C9D3DC" };
function cell(children, width, o = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    borders: { top: thin, bottom: thin, left: thin, right: thin },
    ...o,
    children: Array.isArray(children) ? children : [children],
  });
}
function table(widths, header, rows) {
  const total = widths.reduce((a, b) => a + b, 0);
  const mk = (cells, isHead) =>
    new TableRow({
      tableHeader: isHead,
      cantSplit: true,
      children: cells.map((c, i) =>
        cell(
          p([t(c, { size: 17, bold: isHead || i === 0, color: isHead ? "FFFFFF" : NAVY })], { spacing: { after: 0, line: 240 } }),
          widths[i],
          { shading: { type: ShadingType.CLEAR, fill: isHead ? NAVY : i === 0 ? LIGHT : "FFFFFF", color: "auto" } }
        )
      ),
    });
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows: [mk(header, true), ...rows.map((r) => mk(r, false))] });
}

// KPI tiles
function kpis(items) {
  const w = Math.floor(W / items.length);
  return new Table({
    width: { size: w * items.length, type: WidthType.DXA },
    columnWidths: items.map(() => w),
    rows: [
      new TableRow({
        children: items.map(([big, small, tag]) =>
          new TableCell({
            width: { size: w, type: WidthType.DXA },
            margins: { top: 100, bottom: 100, left: 110, right: 110 },
            shading: { type: ShadingType.CLEAR, fill: LIGHT, color: "auto" },
            borders: { top: { style: BorderStyle.SINGLE, size: 18, color: TEAL }, bottom: none, left: { style: BorderStyle.SINGLE, size: 12, color: "FFFFFF" }, right: { style: BorderStyle.SINGLE, size: 12, color: "FFFFFF" } },
            children: [
              p([new TextRun({ text: big, font: FONT, size: 40, bold: true, color: NAVY })], { spacing: { after: 20 } }),
              p([t(small, { size: 17, color: NAVY })], { spacing: { after: 20, line: 240 } }),
              p([t(tag, { size: 14, color: GREY, italics: true })], { spacing: { after: 0 } }),
            ],
          })
        ),
      }),
    ],
  });
}

const children = [
  // ---------- title block ----------
  new Paragraph({ spacing: { after: 20 }, children: [new TextRun({ text: "AEROMIND", font: FONT, size: 22, bold: true, color: TEAL, characterSpacing: 60 })] }),
  new Paragraph({ spacing: { after: 40 }, children: [new TextRun({ text: "Real-time onboard AI for predictive aircraft health and maintenance", font: FONT, size: 40, bold: true, color: NAVY })] }),
  p([t("Pre-read  |  Tata InnoVent 2027  |  Project FY27-605244  |  Category: AI at the Edge Solutions for Aerospace", { size: 18, color: GREY })], { spacing: { after: 40 } }),
  p([t("Team AeroMind  |  Team leader: Animesh  |  Code, data and tests: github.com/animesh8787/aeromind", { size: 18, color: GREY })], { spacing: { after: 140 } }),

  // ---------- exec summary ----------
  h1("1. Summary"),
  p("AeroMind runs on the aircraft. It reads six sensor families once per second, decides whether a component is degrading, names the fault, estimates remaining useful life with an honest uncertainty band, explains the call with physics, and sends the ground one short message instead of the raw data."),
  p("This is a working proof of concept: a live ground-station demo, a tested Python code base and an evidence report. It runs on simulated data plus two public datasets (NASA IMS bearings and NASA C-MAPSS). It has not flown, and it has not run on Jetson hardware. This document says which result comes from which source."),
  new Paragraph({ spacing: { after: 60 }, children: [] }),
  kpis([
    ["74.8 h", "warning before a real bearing failed (NASA IMS)", "REAL DATA, one test rig"],
    ["0", "false advisories per 1,000 healthy windows", "SIMULATOR"],
    ["545x", "less downlink than raw data; every advisory fits one ACARS block", "SIMULATOR"],
    ["1.2 ms", "per 1 s window on one CPU thread; 1.8 MB of models", "CPU, NOT JETSON"],
  ]),

  // ---------- problem ----------
  h1("2. The problem"),
  bullet([bold("Maintenance is scheduled or reactive. "), t("Progressive degradation is often found after the flight, by an inspection or an alarm that fires too late.")]),
  bullet([bold("Late discovery is expensive. "), t("Aircraft-on-ground (AOG) time is commonly estimated at $150k per hour or more. This is an industry-style assumption, not a measurement of ours, and the ROI section varies it down to $10k.")]),
  bullet([bold("The data cannot all go to the ground. "), t("In the simulator one 1-second snapshot of the six sensor families is 10.5 KB of float32 samples, far more than an ACARS block (220 characters in our format) can carry. Streaming it all is not realistic, so AeroMind sends only advisories.")]),
  bullet([bold("Static thresholds do not learn. "), t("They cannot tell a take-off transient from a fault, or a failed probe from a failed part.")]),

  // ---------- solution ----------
  h1("3. The solution"),
  p("Models run on the aircraft with no connectivity needed. The ground station receives only advisories, then plans maintenance and manages the fleet."),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: [png("architecture.png", 520)] }),
  note("Figure 1. Edge pipeline and ground station. Anomaly score = IsolationForest + autoencoder; classifier and quantile RUL = gradient-boosted trees (an LSTM is also available); RUL interval = conformalized quantile regression."),

  // ---------- differentiators ----------
  h1("4. What makes it different"),
  table([2500, 5000, 2966], ["Capability", "What it does", "Evidence"], [
    ["Flight-phase awareness", "Conditions on taxi, take-off, climb, cruise, descent and outside temperature, so a normal take-off is not an alarm.", "Anomaly gate open on 0.0% of healthy windows, vs 98.6% without phase awareness (simulator)"],
    ["Bad sensor vs bad part", "Seven kinds of sensor check (flatline, stuck, bias, spike, range, rate, NaN). Faulty channels are masked and reported as a sensor fault.", "0 false component advisories on 7 injected sensor failures (up to 38 without); 0 false sensor faults in 5,400 clean windows"],
    ["Explains itself with physics", "Envelope spectrum compared with bearing defect frequencies (BPFO, BPFI, BSF) and current THD, attached to the advisory.", "BPFO line identifies the failing real bearing in 98.9% of IMS snapshots"],
    ["Honest uncertainty", "RUL is given as p10 / p50 / p90 and calibrated with conformal prediction.", "Coverage 0.74 to 0.99 (simulator); 0.71 to 0.87 (C-MAPSS); nominal 0.80"],
    ["Decision, not just a score", "Probability of failure before the next check maps to GROUND NOW, REPLACE AT NEXT CHECK or DEFER AND MONITOR, with a work order.", "Demonstrated in the ground station"],
    ["Fleet learning and secure updates", "Federated averaging shares rare faults without sharing raw data. Model updates are Ed25519-signed, with SHA-256 manifests and rollback.", "91.3% on fault types never seen locally (local only: 0%). A tampered model is rejected in the demo"],
  ]),

  // ---------- evidence ----------
  h1("5. Evidence"),
  p("Everything below is regenerated by one command, python -m aeromind report, which writes docs/evidence-report.md. The Data column says where each number comes from."),
  table([3300, 3400, 2000, 1766], ["Result", "Measured value", "Data", "Caveat"], [
    ["Early warning, real bearing", "Gate opened 74.8 h before end of test; BPFO evidence 74.5 h before", "Real: NASA IMS test 2", "One rig, one failed bearing"],
    ["Early warning, 5 fault types", "Detected 30/30 runs; median lead 74.8 to 163.0 h", "Simulated", "Synthetic faults"],
    ["Names the fault", "First classified alert correct in 29/30 runs (oil contamination 5/6)", "Simulated", "Synthetic faults"],
    ["RUL, NASA C-MAPSS (RMSE, FD001 to FD004)", "LSTM 16.0 / 14.6 / 16.1 / 15.8; trees 19.4 / 17.7 / 22.3 / 20.1; constant baseline 48.5 to 62.9", "Public benchmark", "Turbofan data, not our hardware"],
    ["False advisories", "0.0 per 1,000 healthy windows", "Simulated", ""],
    ["Downlink", "545x less than raw; ACARS max 141 / mean 105.4 of 220 characters; 68 of 68 messages fit", "Simulated", ""],
    ["Compute", "1.21 ms per window mean; 1.8 MB models; INT8 gave no benefit", "Measured on a CPU", "Not Jetson"],
    ["Federated learning", "91.3% on unseen-locally fault types vs 0% local-only (3 seeds)", "Simulated", ""],
  ]),
  new Paragraph({ spacing: { after: 80 }, children: [] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 30 }, children: [png("cmapss_rmse.png", 400)] }),
  note("Figure 2. RUL error on the official NASA C-MAPSS test split (lower is better). Intervals are conformalized."),

  // ---------- business ----------
  h1("6. Business case"),
  p("A fleet simulation (30 aircraft x 3,000 flight hours, common random numbers) compares three policies. Costs and failure rates are assumptions, and the sensitivity table shows how much they matter."),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: [png("roi.png", 400)] }),
  table([5200, 2633, 2633], ["Scenario", "AeroMind vs reactive", "Fixed interval vs reactive"], [
    ["As measured on the simulator", "-98.6%", "-24.0%"],
    ["Detection 67%, warning x0.1, 5 false advisories per 1,000 windows", "-60.4%", "-24.0%"],
    ["As measured, AOG cost $10k/h instead of $150k/h", "-84.2%", "+10.4%"],
    ["Harshest case: detection 67%, warning x0.1, 5 FA/1,000, AOG $10k/h", "-34.4%", "+10.4%"],
  ]),
  note("The simulation is not a forecast. The honest range is -34% to -99% under stated assumptions; real savings need real failure data."),
  p([bold("Fit. "), t("AeroMind is hardware-agnostic and edge-native, so it suits airline MRO and OEM programmes. The design is privacy-first: raw sensor data stays on the aircraft, and ground-side advisories can feed existing MRO and planning systems.")]),

  // ---------- demo ----------
  h1("7. Live demo"),
  p("The judges will see a ground station for six simulated aircraft (VT-AMA01 to VT-AMA06), flying simulated flights:"),
  bullet("Inject a bearing fault. The anomaly score rises, the persistence gate opens, the fault is named, the RUL band falls, and the evidence line shows the envelope peak at the BPFO."),
  bullet("Inject a temperature sensor failure. The channel is masked and a sensor fault is raised, but no component alarm."),
  bullet("Push a signed model update (accepted), a tampered one (rejected, fleet unchanged), then roll back."),
  bullet("Open the fleet ROI panel and the what-if slider for the decision engine."),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 40 }, children: [png("ground-station.png", 520)] }),
  note("Figure 3. Ground station (simulated aircraft and data)."),

  // ---------- limits ----------
  h1("8. Limitations and roadmap"),
  table([5233, 5233], ["Limitations (stated openly)", "Roadmap"], [
    ["No real flight data, no flight or hardware-in-the-loop test, no regulatory work. Simulator faults are synthetic; IMS is one rig.", "Next 3 months: validate on more public run-to-failure sets (CWRU, N-CMAPSS) and tune thresholds on partner data."],
    ["Jetson and TensorRT export is prepared (Hummingbird tensor graphs, build scripts) but not built or run on hardware. All timings are CPU.", "3 to 6 months: Jetson bench validation with replayed sensor data; measure latency, memory and power."],
    ["CWRU data was blocked by the build environment (HTTP 403). N-CMAPSS (15.8 GB) was not run.", "6 to 12 months: hardware-in-the-loop on a test rig, then a shadow-mode trial beside an existing HUMS."],
    ["The ground-side LLM maintenance copilot is not built (it needs an API key). ROI costs and rates are assumptions.", "Alongside: MRO connectors, the LLM copilot for work-order text, and a regulatory-alignment study."],
  ]),
];

const doc = new Document({
  creator: "Team AeroMind",
  title: "AeroMind pre-read",
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  numbering: { config: [{ reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 240 } } } }] }] },
  sections: [
    {
      properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 900, bottom: 800, left: 720, right: 720, header: 400, footer: 400 } } },
      headers: { default: new Header({ children: [new Paragraph({ tabStops: [{ type: TabStopType.RIGHT, position: W }], children: [new TextRun({ text: "AeroMind  |  Tata InnoVent 2027  |  FY27-605244", font: FONT, size: 16, color: GREY }), new TextRun({ text: "\tPre-read", font: FONT, size: 16, color: GREY })] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({ tabStops: [{ type: TabStopType.RIGHT, position: W }], children: [new TextRun({ text: "Proof of concept on simulated and public data. Not flight-tested. Not validated on Jetson hardware.", font: FONT, size: 15, color: GREY }), new TextRun({ text: "\tPage ", font: FONT, size: 15, color: GREY }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 15, color: GREY }), new TextRun({ text: " of ", font: FONT, size: 15, color: GREY }), new TextRun({ children: [PageNumber.TOTAL_PAGES], font: FONT, size: 15, color: GREY })] })] }) },
      children,
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(ROOT, "docs/preread/AeroMind-PreRead.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out);
});
