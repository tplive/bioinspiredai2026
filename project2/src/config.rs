use serde::Deserialize;
use std::path::Path;

// ── Full configuration ────────────────────────────────────────────────────────

/// All tunable hyper-parameters for one GA run.
///
/// Sources are applied in priority order (highest last):
///   1. Built-in defaults (`Config::default()`)
///   2. TOML configuration file (`--config`)
///   3. Individual CLI flags
#[derive(Debug, Clone)]
pub struct Config {
    /// Problem instance JSON file path.
    pub file: String,
    /// Number of individuals in the population.
    pub pop_size: usize,
    /// Maximum number of generations to run.
    pub generations: usize,
    /// Fraction of the population forwarded to the parent pool [0.0–1.0].
    pub selection_ratio: f64,
    /// Probability that crossover is applied to each parent pair [0.0–1.0].
    pub crossover_rate: f64,
    /// Probability that any given individual is mutated [0.0–1.0].
    pub mutation_rate: f64,
    /// Intra-route mutation operator: `"swap"` or `"insert"`.
    pub mutation_type: String,
    /// Fraction of offspring copied into the next generation [0.0–1.0].
    pub reinsertion_ratio: f64,
    /// Multiplier applied to each unit of constraint violation.
    pub penalty_factor: f64,
    /// Population initialisation method: `"random"`, `"nn"` (nearest-neighbour), or `"cw"` (clarke-wright).
    pub init: String,
    /// Selection method: `"truncation"` or `"tournament"`.
    pub selection_type: String,
    /// Tournament size for tournament selection (typically 2-5). Ignored if selection_type is "truncation".
    pub tournament_size: usize,
    /// Write a fitness-history PNG plot after the run. Output is named `<instance>_fitness.png`.
    pub plot: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            file: "train/train_0.json".to_string(),
            pop_size: 100,
            generations: 500,
            selection_ratio: 0.80,
            crossover_rate: 0.85,
            mutation_rate: 0.10,
            mutation_type: "swap".to_string(),
            reinsertion_ratio: 0.85,
            penalty_factor: 10.0,
            init: "random".to_string(),
            selection_type: "tournament".to_string(),
            tournament_size: 3,
            plot: false,
        }
    }
}

// ── Partial configuration (used for TOML and CLI overlays) ───────────────────

/// Mirrors `Config` but every field is optional.
///
/// This allows a TOML file or a set of CLI flags to specify only the fields
/// they care about; absent fields fall through to the layer below.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct PartialConfig {
    pub file: Option<String>,
    pub pop_size: Option<usize>,
    pub generations: Option<usize>,
    pub selection_ratio: Option<f64>,
    pub crossover_rate: Option<f64>,
    pub mutation_rate: Option<f64>,
    pub mutation_type: Option<String>,
    pub reinsertion_ratio: Option<f64>,
    pub penalty_factor: Option<f64>,
    pub init: Option<String>,
    pub selection_type: Option<String>,
    pub tournament_size: Option<usize>,
    pub plot: Option<bool>,
}

impl PartialConfig {
    /// Overlay every `Some` value in `self` on top of `base`.
    pub fn apply_onto(self, mut base: Config) -> Config {
        if let Some(v) = self.file               { base.file = v; }
        if let Some(v) = self.pop_size           { base.pop_size = v; }
        if let Some(v) = self.generations        { base.generations = v; }
        if let Some(v) = self.selection_ratio    { base.selection_ratio = v; }
        if let Some(v) = self.crossover_rate     { base.crossover_rate = v; }
        if let Some(v) = self.mutation_rate      { base.mutation_rate = v; }
        if let Some(v) = self.mutation_type      { base.mutation_type = v; }
        if let Some(v) = self.reinsertion_ratio  { base.reinsertion_ratio = v; }
        if let Some(v) = self.penalty_factor     { base.penalty_factor = v; }
        if let Some(v) = self.init               { base.init = v; }
        if let Some(v) = self.selection_type     { base.selection_type = v; }
        if let Some(v) = self.tournament_size    { base.tournament_size = v; }
        if let Some(v) = self.plot               { base.plot = v; }
        base
    }
}

// ── TOML file loader ──────────────────────────────────────────────────────────

/// Read a TOML file and overlay it onto `Config::default()`.
///
/// Any keys that are absent in the file keep their built-in default values,
/// so it is safe to supply only the parameters you want to change.
pub fn load_file(path: &Path) -> Result<Config, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read config '{}': {e}", path.display()))?;
    let partial: PartialConfig = toml::from_str(&text)
        .map_err(|e| format!("cannot parse config '{}': {e}", path.display()))?;
    Ok(partial.apply_onto(Config::default()))
}
