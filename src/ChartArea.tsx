import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

const ChartArea: React.FC = () => {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const { current } = ref;
    if (current) {
      d3.select(current)
        .append('svg')
        .attr('width', 400)
        .attr('height', 400)
        .append('circle')
        .attr('cx', 100)
        .attr('cy', 250)
        .attr('r', 70)
        .attr('fill', 'green');
    }
  }, [ref]);

  return <div ref={ref} />;
};

export default ChartArea;
