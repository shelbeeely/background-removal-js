export { createOnnxSession, runOnnxSession };

import ndarray, { NdArray } from 'ndarray';
import { InferenceSession, Tensor } from 'onnxruntime-web';
import * as caps from './capabilities';
import { loadAsUrl } from './resource';
import { Config } from './schema';

type ORT = typeof import('onnxruntime-web');
type OnnxBackend = 'wasm' | 'webgpu' | 'webnn';
type WebnnExecutionProvider = {
  name: 'webnn';
  deviceType: 'npu';
};
type BackendConfig = {
  executionProviders: Array<string | WebnnExecutionProvider>;
  useJsepWasm: boolean;
  supportsProxyToWorker: boolean;
};

const BACKEND_CONFIGS: Record<OnnxBackend, BackendConfig> = {
  wasm: {
    executionProviders: ['wasm'],
    useJsepWasm: false,
    supportsProxyToWorker: false
  },
  webgpu: {
    executionProviders: ['webgpu'],
    useJsepWasm: true,
    supportsProxyToWorker: true
  },
  webnn: {
    executionProviders: [{ name: 'webnn', deviceType: 'npu' }],
    useJsepWasm: false,
    supportsProxyToWorker: false
  }
};

const ortByBackend = new Map<OnnxBackend, Promise<ORT>>();
const sessionBackends = new WeakMap<InferenceSession, OnnxBackend>();

const resolveBackend = async (config: Config): Promise<OnnxBackend> => {
  switch (config.device) {
    case 'gpu':
      return (await caps.webgpu()) ? 'webgpu' : 'wasm';
    case 'npu':
      if (await caps.webnn()) return 'webnn';
      if (await caps.webgpu()) return 'webgpu';
      return 'wasm';
    default:
      return 'wasm';
  }
};

const getOrt = async (backend: OnnxBackend): Promise<ORT> => {
  let ort = ortByBackend.get(backend);
  if (ort) return ort;

  switch (backend) {
    case 'webgpu':
      ort = import('onnxruntime-web/webgpu').then((mod) => mod.default);
      break;
    case 'webnn':
    case 'wasm':
      ort = import('onnxruntime-web').then((mod) => mod.default);
      break;
  }

  ortByBackend.set(backend, ort);
  return ort;
};

async function createOnnxSession(model: any, config: Config) {
  const backend = await resolveBackend(config);
  const backendConfig = BACKEND_CONFIGS[backend];
  const proxyToWorker =
    backendConfig.supportsProxyToWorker && config.proxyToWorker;
  const ort = await getOrt(backend);

  if (config.debug) {
    console.debug('\tRequested Device:', config.device);
    console.debug('\tResolved Backend:', backend);
    console.debug('\tProxy to Worker:', proxyToWorker);

    ort.env.debug = true;
    ort.env.logLevel = 'verbose';
  }

  if (backendConfig.useJsepWasm) {
    ort.env.wasm.numThreads = caps.maxNumThreads();
    ort.env.wasm.proxy = proxyToWorker;

    const baseFilePath = '/onnxruntime-web/ort-wasm-simd-threaded.jsep';

    const wasmPath = await loadAsUrl(`${baseFilePath}.wasm`, config);
    const mjsPath = await loadAsUrl(`${baseFilePath}.mjs`, config);

    ort.env.wasm.wasmPaths = {
      mjs: mjsPath,
      wasm: wasmPath
    };
  } else if (backend === 'wasm') {
    ort.env.wasm.numThreads = caps.maxNumThreads();
    ort.env.wasm.proxy = false;

    const baseFilePath = '/onnxruntime-web/ort-wasm-simd-threaded';
    const wasmPath = await loadAsUrl(`${baseFilePath}.wasm`, config);
    const mjsPath = await loadAsUrl(`${baseFilePath}.mjs`, config);

    ort.env.wasm.wasmPaths = {
      mjs: mjsPath,
      wasm: wasmPath
    };
  }

  if (config.debug) {
    console.debug('ort.env.wasm:', ort.env.wasm);
  }

  const ortConfig: InferenceSession.SessionOptions = {
    executionProviders:
      backendConfig.executionProviders as InferenceSession.SessionOptions['executionProviders'],
    graphOptimizationLevel: 'all',
    executionMode: 'parallel',
    enableCpuMemArena: true
  };

  const session = await ort.InferenceSession.create(model, ortConfig).catch(
    (e: any) => {
      throw new Error(
        `Failed to create session: "${e}". Please check if the publicPath is set correctly.`
      );
    }
  );
  sessionBackends.set(session, backend);
  return session;
}

async function runOnnxSession(
  session: any,
  inputs: [string, NdArray<Float32Array>][],
  outputs: [string],
  config: Config
) {
  const backend =
    sessionBackends.get(session) ?? (await resolveBackend(config));
  const ort = await getOrt(backend);

  const feeds: Record<string, any> = {};
  for (const [key, tensor] of inputs) {
    feeds[key] = new ort.Tensor(
      'float32',
      new Float32Array(tensor.data),
      tensor.shape
    );
  }
  const outputData = await session.run(feeds, {});
  const outputKVPairs: NdArray<Float32Array>[] = [];
  for (const key of outputs) {
    const output: Tensor = outputData[key];
    const shape: number[] = output.dims as number[];
    const data: Float32Array = output.data as Float32Array;
    const tensor = ndarray(data, shape);
    outputKVPairs.push(tensor);
  }

  return outputKVPairs;
}
