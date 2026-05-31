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

        baseOverrides = self: super: {
          aim-ui = self.callPackage ./nix/packages/aim-ui.nix { };
          aimrecords = self.callPackage ./nix/packages/aimrecords.nix { };
          aimrocks = self.callPackage ./nix/packages/aimrocks.nix {
            inherit (pkgs) autoPatchelfHook stdenv zlib;
          };
          aim = self.callPackage ./nix/packages/aim.nix { aimUi = self.aim-ui; };

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

        pythonEnv = mkPythonEnv pythonCPU;

        check = pkgs.writeShellApplication {
          name = "check";
          runtimeInputs = [
            pythonEnv
            pkgs.ty
          ];
          text = # bash
            ''
              export PYTHONPATH=src
              exec ty check "$@"
            '';
        };

        packages = with pkgs; [
          ty
          texliveFull
          graphviz

          check
        ];

        shellHook = # bash
          ''
            export PYTHONPATH=src
            export SOURCE_DATE_EPOCH=$(date +%s)
          '';
      in
      {
        apps = {
          check = {
            type = "app";
            program = "${check}/bin/check";
            meta.description = "Run ty check with the Python environment";
          };
        };

        devShells = {
          default = pkgs.mkShell {
            buildInputs = [ pythonEnv ] ++ packages;
            inherit shellHook;
          };

          cuda = pkgs.mkShell {
            buildInputs = [ (mkPythonEnv pythonCUDA) ] ++ packages;

            shellHook =
              let
                libPath = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ];
              in
              # bash
              ''
                export LD_LIBRARY_PATH="${libPath}:/run/opengl-driver/lib:/run/opengl-driver-32/lib:$LD_LIBRARY_PATH"
                ${shellHook} 
              '';
          };
        };
      }
    );
}
