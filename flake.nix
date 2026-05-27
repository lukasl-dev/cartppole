{
  description = "cartPPOle — PPO implementation for CartPole";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { nixpkgs, utils, ... }:
    utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        gymnasium =
          ps:
          ps.gymnasium.overridePythonAttrs (_: {
            # the nixpkgs gymnasium check suite pulls in jax/tensorflow/mujoco/etc.
            # we only need the cartpole runtime dependencies here.
            doCheck = false;
            nativeCheckInputs = [ ];
          });

        # keep the default direnv shell cpu-only and small-ish. cartpole does
        # not need cuda, and enabling cuda in nixpkgs can pull a very large
        # dependency graph that may exhaust ram/disk during `direnv allow`.
        pythonEnv = pkgs.python312.withPackages (
          ps: with ps; [
            torch
            (gymnasium ps)
            numpy
            matplotlib
            ipython
            ruff
          ]
        );

        pythonCudaEnv = pkgs.python312.withPackages (
          ps: with ps; [
            torch-bin
            (gymnasium ps)
            numpy
            matplotlib
            ipython
            ruff
          ]
        );
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [ pythonEnv ];
        };

        devShells.cuda = pkgs.mkShell {
          buildInputs = [ pythonCudaEnv ];

          shellHook =
            let
              libraryPath = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ];
            in
            ''
              export LD_LIBRARY_PATH="${libraryPath}:/run/opengl-driver/lib:/run/opengl-driver-32/lib:$LD_LIBRARY_PATH"
            '';
        };
      }
    );
}
