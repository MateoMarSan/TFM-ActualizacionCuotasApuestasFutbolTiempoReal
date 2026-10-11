"""Despliegue completo en AWS: bucket, zips de las Lambdas y del modelo, pila
de la arquitectura (o actualización si ya existe) y partidos de test en
DynamoDB.

python infra/desplegar.py      (para quitarlo todo: python infra/borrar.py)
"""

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile

import boto3
import torch

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFRA = os.path.join(RAIZ, 'infra')
DIST = os.path.join(INFRA, 'dist')
REGION = boto3.session.Session().region_name or 'us-east-1'
IMAGEN = (f'763104351884.dkr.ecr.{REGION}.amazonaws.com/'
          'pytorch-inference:2.6.0-cpu-py312-ubuntu22.04-sagemaker')
NUMPY_LAMBDA = 'numpy==2.2.6'      # versión para la capa de Lambda (Python 3.12, Linux)

# Código de cada Lambda: su handler + los módulos compartidos con el entrenamiento
LAMBDAS = {
    'procesador': ['infra/lambdas/procesador/handler.py', 'procesado.py'],
    'cuotas': ['infra/lambdas/cuotas/handler.py', 'odds_generator.py', 'procesado.py'],
    'plataforma': ['infra/lambdas/plataforma/handler.py'],
    'api': ['infra/lambdas/api/handler.py', 'infra/lambdas/api/index.html'],
}


def aws_cli() -> str:
    ruta = shutil.which('aws') or r'C:\Program Files\Amazon\AWSCLIV2\aws.exe'
    if not os.path.exists(ruta) and not shutil.which('aws'):
        sys.exit('No encuentro la AWS CLI')
    return ruta


def desplegar_plantilla(plantilla: str, pila: str, parametros: dict = None):
    # Una pila cuya primera creación falló queda en ROLLBACK_COMPLETE y no se
    # puede actualizar: hay que borrarla antes de volver a intentarlo
    cfn = boto3.client('cloudformation')
    try:
        estado = cfn.describe_stacks(StackName=pila)['Stacks'][0]['StackStatus']
    except cfn.exceptions.ClientError:
        estado = None
    if estado == 'ROLLBACK_COMPLETE':
        print(f'  {pila} quedó en ROLLBACK_COMPLETE tras un fallo anterior: se borra')
        cfn.delete_stack(StackName=pila)
        cfn.get_waiter('stack_delete_complete').wait(StackName=pila)
    cmd = [aws_cli(), 'cloudformation', 'deploy', '--template-file',
           os.path.join(INFRA, plantilla), '--stack-name', pila,
           '--capabilities', 'CAPABILITY_IAM', '--no-fail-on-empty-changeset']
    if parametros:
        cmd += ['--parameter-overrides'] + [f'{k}={v}' for k, v in parametros.items()]
    print(f'\nDesplegando {pila} ...')
    subprocess.run(cmd, check=True)


def salida(pila: str, clave: str) -> str:
    outs = boto3.client('cloudformation').describe_stacks(StackName=pila)['Stacks'][0]['Outputs']
    return next(o['OutputValue'] for o in outs if o['OutputKey'] == clave)


def subir(bucket: str, ruta_local: str, nombre: str, huella: str = None) -> str:
    """Sube un artefacto con su hash en el nombre: si el contenido cambia, cambia
    la clave y CloudFormation actualiza el recurso que lo usa. `huella` permite
    dar un hash calculado a partir de las fuentes (para ficheros no deterministas)."""
    if huella is None:
        with open(ruta_local, 'rb') as f:
            huella = hashlib.sha256(f.read()).hexdigest()
    h = huella[:12]
    extension = '.tar.gz' if ruta_local.endswith('.tar.gz') else os.path.splitext(ruta_local)[1]
    clave = f'artefactos/{nombre}-{h}{extension}'
    s3 = boto3.client('s3')
    try:
        s3.head_object(Bucket=bucket, Key=clave)
    except s3.exceptions.ClientError:
        print(f'  subiendo {clave}')
        s3.upload_file(ruta_local, bucket, clave)
    return clave


def zip_determinista(destino: str, archivos: dict):
    """Zip con fechas fijas: el mismo código produce siempre el mismo hash."""
    with zipfile.ZipFile(destino, 'w', zipfile.ZIP_DEFLATED) as z:
        for nombre, ruta in sorted(archivos.items()):
            info = zipfile.ZipInfo(nombre, date_time=(2026, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            with open(ruta, 'rb') as f:
                z.writestr(info, f.read())


def capa_numpy() -> str:
    """NumPy compilado para Linux y Python 3.12 (el entorno de Lambda)."""
    destino = os.path.join(DIST, 'capa-numpy.zip')
    if os.path.exists(destino):
        return destino
    ruedas = os.path.join(DIST, 'ruedas')
    subprocess.run([sys.executable, '-m', 'pip', 'download', NUMPY_LAMBDA, '--only-binary=:all:',
                    '--platform', 'manylinux2014_x86_64', '--python-version', '3.12',
                    '--implementation', 'cp', '-d', ruedas, '-q'], check=True)
    rueda = next(os.path.join(ruedas, f) for f in os.listdir(ruedas) if f.endswith('.whl'))
    archivos = {}
    with zipfile.ZipFile(rueda) as w:
        extraer = os.path.join(DIST, 'capa')
        w.extractall(extraer)
    for carpeta, _, ficheros in os.walk(extraer):
        for f in ficheros:
            ruta = os.path.join(carpeta, f)
            archivos['python/' + os.path.relpath(ruta, extraer).replace(os.sep, '/')] = ruta
    zip_determinista(destino, archivos)
    return destino


def paquete_modelo() -> tuple[str, float, str]:
    """model.tar.gz para SageMaker: pesos + código de inferencia. Devuelve también
    una huella del checkpoint y del código: torch.save no produce siempre los
    mismos bytes, así que el hash del .tar.gz cambiaría aunque nada hubiera cambiado."""
    ruta_ckpt = os.path.join(RAIZ, 'checkpoints', 'best_model.pt')
    fuentes = [ruta_ckpt, os.path.join(INFRA, 'modelo', 'inference.py'),
               os.path.join(RAIZ, 'axial_transformer.py'), os.path.join(RAIZ, 'procesado.py')]
    huella = hashlib.sha256(b''.join(open(f, 'rb').read() for f in fuentes)).hexdigest()
    ckpt = torch.load(ruta_ckpt, map_location='cpu', weights_only=False)
    temperatura = float(ckpt.get('temperature', 1.0))
    buffer = io.BytesIO()
    torch.save({'model_state_dict': ckpt['model_state_dict'], 'args': ckpt['args']}, buffer)
    destino = os.path.join(DIST, 'model.tar.gz')
    contenido = {
        'model.pth': buffer.getvalue(),
        'code/inference.py': open(os.path.join(INFRA, 'modelo', 'inference.py'), 'rb').read(),
        'code/axial_transformer.py': open(os.path.join(RAIZ, 'axial_transformer.py'), 'rb').read(),
        'code/procesado.py': open(os.path.join(RAIZ, 'procesado.py'), 'rb').read(),
    }
    with tarfile.open(destino, 'w:gz', format=tarfile.GNU_FORMAT) as tar:
        for nombre, datos in sorted(contenido.items()):
            info = tarfile.TarInfo(nombre)
            info.size, info.mtime, info.mode = len(datos), 0, 0o644
            tar.addfile(info, io.BytesIO(datos))
    return destino, temperatura, huella


def main():
    os.makedirs(DIST, exist_ok=True)
    print(f'Cuenta: {boto3.client("sts").get_caller_identity()["Arn"]}  |  región {REGION}')

    desplegar_plantilla('base.yaml', 'tfm-base')
    bucket = salida('tfm-base', 'BucketArtefactos')

    print('\nEmpaquetando y subiendo artefactos ...')
    claves = {'ClaveCapaNumpy': subir(bucket, capa_numpy(), 'capa-numpy')}
    for nombre, ficheros in LAMBDAS.items():
        zip_local = os.path.join(DIST, f'lambda-{nombre}.zip')
        zip_determinista(zip_local, {os.path.basename(f): os.path.join(RAIZ, f) for f in ficheros})
        claves[f'Clave{nombre.capitalize()}'] = subir(bucket, zip_local, f'lambda-{nombre}')
    modelo, temperatura, huella = paquete_modelo()
    claves['ClaveModelo'] = subir(bucket, modelo, 'model', huella)

    desplegar_plantilla('arquitectura.yaml', 'tfm-arquitectura', {
        'BucketArtefactos': bucket, 'ImagenInferencia': IMAGEN,
        'Temperatura': f'{temperatura:.4f}', **claves})

    print('\nPreparando los partidos de test (nivel previo en DynamoDB) ...')
    subprocess.run([sys.executable, os.path.join(INFRA, 'preparar_partidos.py')], check=True)

    url = salida('tfm-arquitectura', 'UrlWeb')
    with open(os.path.join(DIST, 'despliegue.json'), 'w', encoding='utf-8') as f:
        json.dump({'url_web': url, 'bucket': bucket, 'region': REGION, **claves}, f, indent=1)
    print(f'\nDesplegado. Web de la plataforma: {url}')
    print('   Simula un partido con:  python infra/simulador.py')
    print('   Al terminar, bórralo todo con:  python infra/borrar.py')


if __name__ == '__main__':
    main()
