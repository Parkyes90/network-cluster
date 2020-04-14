import React from 'react';
import ChartArea from './ChartArea';

const data = [25, 20, 10, 12, 15];

function App(): JSX.Element {
  return (
    <div>
      <ChartArea data={data} />
    </div>
  );
}

export default App;
