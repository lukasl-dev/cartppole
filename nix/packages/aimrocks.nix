{
  autoPatchelfHook,
  buildPythonPackage,
  fetchPypi,
  stdenv,
  zlib,
}:

buildPythonPackage rec {
  pname = "aimrocks";
  version = "0.5.2";
  format = "wheel";

  nativeBuildInputs = [ autoPatchelfHook ];
  buildInputs = [
    stdenv.cc.cc.lib
    zlib
  ];

  src = fetchPypi {
    inherit pname version format;
    dist = "cp312";
    python = "cp312";
    abi = "cp312";
    platform = "manylinux_2_17_x86_64.manylinux2014_x86_64";
    hash = "sha256-qbpkfzKTSsmZxBGcu4tZUQ3+aa7JhVhTm4TbfbnyCs8=";
  };

  doCheck = false;
}
