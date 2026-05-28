{
  buildPythonPackage,
  fetchPypi,
  setuptools,
}:

buildPythonPackage rec {
  pname = "aim-ui";
  version = "3.29.1";
  pyproject = true;

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-vDAnisrQSKCH+E4/biVl9T916H2MKXSMkEIOwGuCARM=";
  };

  build-system = [ setuptools ];
  doCheck = false;
}
