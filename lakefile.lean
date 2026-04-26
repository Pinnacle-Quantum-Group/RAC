import Lake
open Lake DSL

package «racFormalProofs» where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.5.0"

@[default_target]
lean_lib «RACFormalProofs» where
  srcDir := "formal_proofs"
  globs := #[
    .submodules `Composite,
    .submodules `Cordic,
    .submodules `DSP,
    .submodules `Distributed,
    .submodules `Hardware,
    .submodules `HyperbolicCordic,
    .submodules `Numerical,
    .submodules `Physics,
    .submodules `TableLookup,
    .submodules `Transformer
  ]
