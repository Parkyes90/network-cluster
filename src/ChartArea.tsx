import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface ChartAreaProps {
  data: number[];
}

const ChartArea: React.FC<ChartAreaProps> = ({ data }) => {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const { current } = ref;
    if (current) {
      const svg = d3
        .select(current)
        .append('svg')
        .attr('width', 400)
        .attr('height', 400);

      const circles = svg.selectAll('circle').data(data);
      circles
        .enter()
        .append('circle')
        .attr('cx', (d, i): number => {
          return i * 50 + 25;
        })
        .attr('cy', 250)
        .attr('r', (d): number => {
          return d;
        })
        .attr('fill', 'green');
    }
  }, [ref, data]);

  return <div ref={ref} />;
};

export default ChartArea;
