"""Check OpenUSD collision authoring and imported scene coordinates."""
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from isaac_r2r.cli import convert_scan_asset
from isaac_r2r.scans import add_static_collisions

try:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
except ImportError:
    Gf = Usd = UsdGeom = UsdPhysics = None


@unittest.skipIf(Usd is None, 'usd-core is not installed')
class ScanConversionTest(unittest.TestCase):
    def test_y_up_geometry_rotates_to_z_up_and_retains_static_collision(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, '/World')
        root.AddTranslateOp().Set(Gf.Vec3d(0, 1, 0))
        mesh = UsdGeom.Mesh.Define(stage, '/World/Floor')
        mesh.CreatePointsAttr([Gf.Vec3f(0, 1, 0), Gf.Vec3f(1, 1, 0),
                               Gf.Vec3f(0, 1, 1)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])

        result = add_static_collisions(stage)

        self.assertEqual(result, {'meshes': 1, 'vertices': 3, 'triangles': 1,
                                  'instances_deinstanced': 0,
                                  'source_up_axis': 'Y'})
        self.assertEqual(UsdGeom.GetStageUpAxis(stage), UsdGeom.Tokens.z)
        self.assertTrue(mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI))
        self.assertEqual(UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get(),
                         'none')
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim())
        point = matrix.Transform(Gf.Vec3d(0, 1, 0))
        self.assertAlmostEqual(point[0], 0.0)
        self.assertAlmostEqual(point[1], 0.0)
        self.assertAlmostEqual(point[2], 2.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scan.usda'
            stage.GetRootLayer().Export(str(path))
            reopened = Usd.Stage.Open(str(path))
            self.assertEqual(UsdGeom.GetStageUpAxis(reopened), UsdGeom.Tokens.z)
            self.assertTrue(reopened.GetPrimAtPath('/World/Floor').HasAPI(
                UsdPhysics.CollisionAPI))

    def test_rejects_importer_that_only_creates_empty_mesh_prims(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        mesh = UsdGeom.Mesh.Define(stage, '/World/Empty')

        with self.assertRaisesRegex(ValueError, 'missing or invalid geometry'):
            add_static_collisions(stage)
        self.assertFalse(mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI))

    def test_expands_referenced_instances_before_applying_collision(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, '/World/Asset')
        mesh = UsdGeom.Mesh.Define(stage, '/World/Asset/Mesh')
        mesh.CreatePointsAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0),
                               Gf.Vec3f(0, 1, 0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        for name in ('A', 'B'):
            instance = UsdGeom.Xform.Define(stage, f'/World/{name}').GetPrim()
            instance.GetReferences().AddInternalReference('/World/Asset')
            instance.SetInstanceable(True)
            self.assertTrue(instance.IsInstance())

        result = add_static_collisions(stage)

        self.assertEqual(result['instances_deinstanced'], 2)
        self.assertEqual(result['meshes'], 3)
        self.assertEqual(result['triangles'], 3)
        for name in ('A', 'B'):
            self.assertTrue(stage.GetPrimAtPath(f'/World/{name}/Mesh').HasAPI(
                UsdPhysics.CollisionAPI))


class ScanImportResultTest(unittest.TestCase):
    def test_missing_verified_artifacts_rejects_zero_exit_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.glb'
            source.write_bytes(b'glTF' + (2).to_bytes(4, 'little')
                               + (12).to_bytes(4, 'little'))
            output = Path(directory) / 'scan.usd'
            with patch('isaac_r2r.cli.subprocess.run') as runner:
                runner.return_value.returncode = 0
                with self.assertRaisesRegex(ValueError, 'no verified USD'):
                    convert_scan_asset(Namespace(glb=source, out=output))

    def test_matching_worker_artifacts_are_accepted(self):
        from bench.artifacts import sha256

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.glb'
            source.write_bytes(b'glTF' + (2).to_bytes(4, 'little')
                               + (12).to_bytes(4, 'little'))
            output = Path(directory) / 'scan.usd'

            def produce_artifacts(*_args, **_kwargs):
                output.write_bytes(b'USD')
                output.with_suffix('.json').write_text(json.dumps({
                    'source_sha256': sha256(source), 'usd': str(output),
                    'meshes': 1, 'triangles': 1,
                }), encoding='utf-8')
                return Namespace(returncode=0)

            with patch('isaac_r2r.cli.subprocess.run', side_effect=produce_artifacts):
                result = convert_scan_asset(Namespace(glb=source, out=output))
            self.assertEqual(result['triangles'], 1)


if __name__ == '__main__':
    unittest.main()
