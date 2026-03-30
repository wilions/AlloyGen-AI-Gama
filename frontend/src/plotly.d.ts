declare module 'react-plotly.js' {
  import { Component } from 'react';
  import Plotly from 'plotly.js-dist-min';

  interface PlotParams {
    data: Plotly.Data[];
    layout?: Partial<Plotly.Layout>;
    config?: Partial<Plotly.Config>;
    style?: React.CSSProperties;
    className?: string;
    onClick?: (event: any) => void;
    onHover?: (event: any) => void;
  }

  class Plot extends Component<PlotParams> {}
  export default Plot;
}

declare module 'plotly.js-dist-min' {
  export * from 'plotly.js';
}
