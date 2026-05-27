{
  description = "cartPPOle — PPO implementation for CartPole";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      nixpkgs,
      utils,
      pyproject-nix,
      ...
    }:
    utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        project = pyproject-nix.lib.project.loadUVPyproject {
          projectRoot = ./.;
        };

        mkPythonEnv =
          python:
          let
            resolved = project.renderers.withPackages { inherit python; };
          in
          python.withPackages (
            ps:
            resolved ps
            ++ (with ps; [
              ipython
              pygame
              ruff
            ])
          );

        baseOverrides = _self: super: {
          gymnasium = super.gymnasium.overridePythonAttrs (_: {
            doCheck = false;
            nativeCheckInputs = [ ];
          });
        };

        pythonCPU = pkgs.python312.override {
          packageOverrides = baseOverrides;
        };

        pythonCUDA = pkgs.python312.override {
          packageOverrides =
            self: super:
            (baseOverrides self super)
            // {
              torch = pkgs.python312Packages.torch-bin;
            };
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [ (mkPythonEnv pythonCPU) ];
        };

        devShells.cuda = pkgs.mkShell {
          buildInputs = [ (mkPythonEnv pythonCUDA) ];

          shellHook =
            let
              libPath = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ];
            in
            # bash
            ''
              export LD_LIBRARY_PATH="${libPath}:/run/opengl-driver/lib:/run/opengl-driver-32/lib:$LD_LIBRARY_PATH"
            '';
        };
      }
    );
}
