interface Navigator {
  gpu?: any;
  ml?: {
    createContext: (...args: any[]) => Promise<any>;
  };
}
