"""Check OpenUSD collision authoring and imported scene coordinates."""
import tempfile
import unittest
from pathlib import Path

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

        self.assertEqual(result, {'meshes': 1, 'source_up_axis': 'Y'})
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


if __name__ == '__main__':
    unittest.main()
