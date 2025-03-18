{
  description = "Emotion detection project";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    {
      self,
      nixpkgs,
    }:
    let
      # Supported systems
      systems = [
        "aarch64-linux"
        "i686-linux"
        "x86_64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];

      # Add the kagglehub package
      overlays = [
        (final: prev: {
          python312 = prev.python312.override {
            packageOverrides = finalPy: prevPy: {
              kagglehub = final.python312.pkgs.buildPythonPackage rec {
                pname = "kagglehub";
                version = "v0.3.10";
                src = final.fetchFromGitHub {
                  owner = "Kaggle";
                  repo = pname;
                  rev = version;
                  sha256 = "sha256-hLktlCRQ00meGmxJNt27qThGzdEzniARvWiDrosrSmM=";
                };

                # Add hatch build system
                build-system = with final.pkgs; [
                  hatch
                ];

                # Add dependencies
                dependencies = with final.python312.pkgs; [
                  requests
                  tqdm
                  packaging
                  pyyaml
                ];

                pyproject = true;
              };
            };
          };
          python312Packages = final.python312.pkgs;
        })
      ];

      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShell = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system overlays;
            config = {
              allowUnfree = true;
              cudaSupport = true;
            };
          };
        in
        pkgs.mkShell {
          buildInputs = with pkgs; [
            # For Numpy, Torch, etc.
            stdenv.cc.cc
            zlib
          ];

          packages = with pkgs; [
            # Build system for loading C++ extensions in torch
            ninja
            cudatoolkit
            gcc13 # Version <=13 required for nvcc

            (python312.withPackages (
              ps: with ps; [
                click
                torch
                torchaudio
                tqdm
                transformers
                pysoundfile
                kagglehub
              ]
            ))
          ];
        }
      );
    };
}
