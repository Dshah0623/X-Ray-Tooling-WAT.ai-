import React, { useEffect, useState } from 'react';
import { Typography, Container, Paper } from '@mui/material';

const classIdToBodyPart = {
  0: 'Abdomen',
  1: 'Ankle',
  2: 'Cervical Spine',
  3: 'Chest',
  4: 'Clavicles',
  5: 'Elbow',
  6: 'Feet',
  7: 'Finger',
  8: 'Forearm',
  9: 'Hand',
  10: 'Hip',
  11: 'Knee',
  12: 'Lower Leg',
  13: 'Lumbar Spine',
  14: 'Others',
  15: 'Pelvis',
  16: 'Shoulder',
  17: 'Sinus',
  18: 'Skull',
  19: 'Thigh',
  20: 'Thoracic Spine',
  21: 'Wrist',
};

const Results = () => {
  const [phaseOneResult, setPhaseOneResult] = useState(null);
  const [phaseTwoResult, setPhaseTwoResult] = useState(null);

    const handleRunPhaseOne = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/phase1', {
        method: 'GET',
      });
      if(response.ok) {
        const data = await response.json();
        console.log('Phase 1 run:', data);
        if (data["class_id"] === 1) {
          data["class_id"] = "Fractured"
        } else {
          data["class_id"] = "Not fractured"
        }
        setPhaseOneResult(data);
      }
    }
    catch (error) {
      console.error('Error running phase 1:', error);
    }

  };  


  const handleRunPhaseTwo = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/phase2', {
        method: 'GET',
      });
      if (response.ok) {
        let data = await response.json();
        console.log('Phase 2 run:', data);
        if (data["class_id"] !== undefined) { // Ensure `class_id` exists in the data
          const bodyPart = classIdToBodyPart[data["class_id"]];
          if (bodyPart) { // Check if the mapping was successful
            data["class_id"] = bodyPart; // Update `class_id` with the body part string
          } else {
            console.error('Invalid class_id received:', data["class_id"]);
          }
        }
        setPhaseTwoResult(data);
      }
    } catch (error) {
      console.error('Error running phase 2:', error);
    }
  };

  useEffect(() => {
    handleRunPhaseOne();
    handleRunPhaseTwo();
  }, []);

  return (
    <Container>
      <Typography variant="h4" gutterBottom>Results</Typography>
      <Paper elevation={3} sx={{ padding: '20px', margin: '20px 0' }}>
        <Typography variant="h6">Phase One Results:</Typography>
        <Typography>{phaseOneResult ? JSON.stringify(phaseOneResult) : 'Loading...'}</Typography>
      </Paper>
      <Paper elevation={3} sx={{ padding: '20px', margin: '20px 0' }}>
        <Typography variant="h6">Phase Two Results:</Typography>
        <Typography>{phaseTwoResult ? JSON.stringify(phaseTwoResult) : 'Loading...'}</Typography>
      </Paper>
    </Container>
  );
};

export default Results;
