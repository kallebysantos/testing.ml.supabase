// Setup type definitions for built-in Supabase Runtime APIs
import "jsr:@supabase/functions-js/edge-runtime.d.ts";

import InferenceAPI from "inference-api";
const { Tensor, RawSession } = InferenceAPI;

// Custom filename on Hugging Face, default: 'model_quantized.onnx'
const session = await RawSession.fromHuggingFace(
  "kallebysantos/vehicle-emission",
  {
    path: {
      modelFile: "model.onnx",
    },
  },
);

Deno.serve(async (req: Request) => {
  const carsBatchInput = await req.json();

  // Parsing objects to tensor input
  const inputTensors = {};
  session.inputs.forEach((inputKey) => {
    // @ts-ignore key index
    const values = carsBatchInput.map((item) => item[inputKey]);

    // This model uses `float32` tensors, but could variate to mixed types
    // @ts-ignore key index
    inputTensors[inputKey] = new Tensor("float32", values, [values.length, 1]);
  });

  const { emissions } = await session.run(inputTensors);

  return Response.json({ result: emissions }); // [ 289.01, 199.53]
});
