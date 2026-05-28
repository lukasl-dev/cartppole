{
  base58,
  buildPythonPackage,
  fetchPypi,
}:

buildPythonPackage rec {
  pname = "aimrecords";
  version = "0.0.7";
  format = "wheel";
  pythonRelaxDeps = [ "base58" ];

  src = fetchPypi {
    inherit pname version format;
    dist = "py2.py3";
    python = "py2.py3";
    abi = "none";
    platform = "any";
    hash = "sha256-uSdokIkcX9aPgX4g/F1GaoDAHiL6Ro6ql5MxRIp11gE=";
  };

  propagatedBuildInputs = [ base58 ];
  doCheck = false;
}
