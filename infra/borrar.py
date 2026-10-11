"""Borra la arquitectura de AWS: las pilas, el registro de modelos y el bucket.
El presupuesto se deja.

python infra/borrar.py
"""

import boto3

cfn = boto3.client('cloudformation')


def existe(pila: str) -> bool:
    try:
        estado = cfn.describe_stacks(StackName=pila)['Stacks'][0]['StackStatus']
        return estado != 'DELETE_COMPLETE'
    except cfn.exceptions.ClientError:
        return False


def borrar_pila(pila: str):
    if not existe(pila):
        print(f'  {pila}: no existe')
        return
    print(f'Borrando {pila} (puede tardar unos minutos) ...')
    cfn.delete_stack(StackName=pila)
    cfn.get_waiter('stack_delete_complete').wait(StackName=pila,
                                                 WaiterConfig={'Delay': 15, 'MaxAttempts': 120})
    print(f'  {pila}: borrada')


def vaciar_bucket(bucket: str):
    """El bucket tiene versionado: hay que borrar todas las versiones."""
    print(f'Vaciando el bucket {bucket} ...')
    boto3.resource('s3').Bucket(bucket).object_versions.delete()


def main():
    borrar_pila('tfm-arquitectura')

    logs = boto3.client('logs')
    for grupo in logs.describe_log_groups(
            logGroupNamePrefix='/aws/sagemaker/Endpoints/tfm-axial-endpoint')['logGroups']:
        logs.delete_log_group(logGroupName=grupo['logGroupName'])
        print(f"  registros {grupo['logGroupName']}: borrados")

    if existe('tfm-base'):
        outs = cfn.describe_stacks(StackName='tfm-base')['Stacks'][0].get('Outputs', [])
        bucket = next((o['OutputValue'] for o in outs if o['OutputKey'] == 'BucketArtefactos'), None)
        if bucket:
            vaciar_bucket(bucket)
    borrar_pila('tfm-base')
    print('\nTodo borrado. El presupuesto (tfm-presupuesto) sigue activo.')


if __name__ == '__main__':
    main()
