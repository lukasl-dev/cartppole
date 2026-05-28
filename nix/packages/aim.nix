{
  aimUi,
  aimrecords,
  aimrocks,
  aiofiles,
  alembic,
  boto3,
  buildPythonPackage,
  cachetools,
  click,
  cryptography,
  fastapi,
  fetchPypi,
  filelock,
  jinja2,
  numpy,
  packaging,
  pillow,
  psutil,
  python-dateutil,
  pytz,
  requests,
  restrictedpython,
  sqlalchemy,
  tqdm,
  uvicorn,
  watchdog,
  websockets,
}:

buildPythonPackage rec {
  pname = "aim";
  version = "3.29.1";
  format = "wheel";

  src = fetchPypi {
    inherit pname version format;
    dist = "cp312";
    python = "cp312";
    abi = "cp312";
    platform = "manylinux_2_28_x86_64";
    hash = "sha256-Fc6gxj2aYYEiKdcUDOwE8CEXYm0XJOEL6+QQVU8/dWE=";
  };

  propagatedBuildInputs = [
    aimUi
    aimrecords
    aimrocks
    aiofiles
    alembic
    boto3
    cachetools
    click
    cryptography
    fastapi
    filelock
    jinja2
    numpy
    packaging
    pillow
    psutil
    python-dateutil
    pytz
    requests
    restrictedpython
    sqlalchemy
    tqdm
    uvicorn
    watchdog
    websockets
  ];

  doCheck = false;
}
