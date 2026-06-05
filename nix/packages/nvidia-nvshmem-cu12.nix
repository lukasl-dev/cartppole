{
  autoPatchelfHook,
  fetchurl,
  stdenv,
  stdenvNoCC,
  unzip,
}:

# nixpkgs' torch-bin uses the PyTorch wheel, but replaces the wheel's
# nvidia-nvshmem-cu12 dependency with cudaPackages.libnvshmem. For the pinned
# nixpkgs revision, that libnvshmem path is not in the CUDA cache, so Nix tries
# to compile it locally. This derivation uses NVIDIA's binary PyPI wheel
# instead; it provides libnvshmem_host.so.3, which is what torch's wheel needs.
stdenvNoCC.mkDerivation {
  pname = "nvidia-nvshmem-cu12";
  version = "3.4.5";

  src = fetchurl {
    url = "https://files.pythonhosted.org/packages/b5/09/6ea3ea725f82e1e76684f0708bbedd871fc96da89945adeba65c3835a64c/nvidia_nvshmem_cu12-3.4.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl";
    hash = "sha256-BC8lAPJMAh24oGxe7CU5An1XRg4cGnYgVaZVT3LDab0=";
  };

  dontUnpack = true;

  nativeBuildInputs = [
    autoPatchelfHook
    unzip
  ];

  buildInputs = [ stdenv.cc.cc.lib ];

  autoPatchelfIgnoreMissingDeps = [
    # Optional NVSHMEM transports/bootstrap plugins that are not needed merely
    # to import/use torch for single-GPU CartPole work.
    "libfabric.so.1"
    "libmpi.so.40"
    "libmlx5.so.1"
    "liboshmem.so.40"
    "libpmix.so.2"
    "libucp.so.0"
    "libucs.so.0"
  ];

  installPhase = ''
    runHook preInstall

    mkdir -p "$out"
    unzip -q "$src" 'nvidia/nvshmem/*' -d wheel
    cp -r wheel/nvidia/nvshmem/include "$out/include"
    cp -r wheel/nvidia/nvshmem/lib "$out/lib"

    runHook postInstall
  '';
}
