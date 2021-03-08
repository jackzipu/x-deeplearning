




import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Identity(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'Identity',
            inputs=['x'],
            outputs=['y'],
        )

        data = np.array([[[
            [1, 2],
            [3, 4],
        ]]], dtype=np.float32)

        expect(node, inputs=[data], outputs=[data],
               name='test_identity')
