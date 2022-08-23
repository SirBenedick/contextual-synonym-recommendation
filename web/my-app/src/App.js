import './App.css';
import { useState } from 'react';
import Container from '@mui/material/Container';
import Typography from '@mui/material/Typography';
import Divider from '@mui/material/Divider';
import CircularProgress from '@mui/material/CircularProgress';
import MyTextEditor from './MyTextEditor'
import { Box } from '@mui/material';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import IconButton from '@mui/material/IconButton';
import ThumbDownIcon from '@mui/icons-material/ThumbDown';
import ThumbUpIcon from '@mui/icons-material/ThumbUp';
import DoneIcon from '@mui/icons-material/Done';
import CloseIcon from '@mui/icons-material/Close';
import { ListItemSecondaryAction } from '@mui/material';
import Tooltip from '@mui/material/Tooltip';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Stack from '@mui/material/Stack';

const App = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [sentenceWithTokenMask, setSentenceWithTokenMask] = useState("");
  const [selectedWord, setSelectedWord] = useState("");
  const [listOfSynonyms, setListOfSynonyms] = useState([]);
  const [err, setErr] = useState('');
  const [synonymSuggestion, setSynonymSuggestion] = useState('');

  const getSynonymsFromAPI = async (masked_sentence, original_token) => {
    setIsLoading(true);
    setListOfSynonyms([])
    try {
      const response = await fetch('http://URL.de/api/predict', {
        method: 'POST',
        body: JSON.stringify({
          masked_sentence: masked_sentence,
          original_token: original_token,
        }),
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Error! status: ${response.status}`);
      }

      const result = await response.json();

      console.log('result is: ', JSON.stringify(result, null, 4));
      const synonyms = result.map((result, index) => { return { word: result, feedbackWasGiven: false, feedbackScore: undefined, rankOfRecommendation: index } })
      setListOfSynonyms(synonyms)

    } catch (err) {
      setErr(err.message);
    } finally {
      setIsLoading(false);
      console.log(err); // leave in, else build won't compile
    }
  };

  const handleFeedback = async (synonym, score) => {
    if (score !== 0 && score !== 1) return false

    try {
      // Send feedback to API
      const response = await fetch('http://URL.de/api/feedback', {
        method: 'POST',
        body: JSON.stringify({
          masked_sentence: sentenceWithTokenMask,
          original_token: selectedWord,
          replacement: synonym.word,
          score: score,
          rank_of_recommendation: synonym.rankOfRecommendation
        }),
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Error! status: ${response.status}`);
      }


      const result = await response.json();

      console.log('Add feedback was successfull?: ', JSON.stringify(result, null, 4));

      // Update listOfSynonyms to display which feedback was given
      const synonymsTemp = listOfSynonyms.map((synonymOfList) => {
        if (synonymOfList.word === synonym.word) return { word: synonym.word, feedbackWasGiven: true, feedbackScore: score, rankOfRecommendation: synonym.rankOfRecommendation }
        else return synonymOfList
      })
      setListOfSynonyms(synonymsTemp)

    } catch (err) {
      setErr(err.message);
    } finally {
      console.log(err); // leave in, else build won't compile
    }

  }

  const suggestSynonym = async () => {
    await handleFeedback({ word: synonymSuggestion }, 1)
    setSynonymSuggestion("")
  }

  return (
    <div className="App">
      <Container maxWidth="md">
        <Box
          display="flex"
          flexDirection="column"
        >
          <Box
            display="flex"
            flexDirection="column"
          >
            <Typography variant="h3" component="div" style={{ padding: "5px" }}>
              Contextual Synonym Recommendation
            </Typography>
            <Divider>Enter a sentence, select a word, receive synonyms</Divider>
          </Box>
          <Box
            display="flex"
            flexDirection="column"
            justifyContent="space-around"
            style={{ minHeight: "100px", padding: "10px" }}
          >
            <MyTextEditor getSynonymsFromAPI={getSynonymsFromAPI}
              setSentenceWithTokenMask={setSentenceWithTokenMask}
              setSelectedWord={setSelectedWord}></MyTextEditor>
          </Box>
          <Box
            style={{ height: "40%" }}
            display="flex"
            flexDirection="column"
            alignItems="center"
          >
            <Typography variant="h6" component="div" style={{ paddingTop: "5px" }}>
              {isLoading ? "Loading Synonyms..." : (listOfSynonyms.length ? `Found Synonyms for "${selectedWord}"` : `No Synonyms found for "${selectedWord}" :(`)}
            </Typography>
            <Typography variant="caption" component="div" style={{ width: "300px" }}>
              Help us improve the recommendations by providing some quick feedback.
              Thank you :)
            </Typography>
            {isLoading ? (<CircularProgress />) : (

              <div>
                <List >
                  {listOfSynonyms.map(synonym => (
                    <ListItem
                      style={{ width: "300px" }}
                    >
                      <ListItemText
                        primary={synonym.word}
                      />
                      {synonym.feedbackWasGiven ? (
                        <ListItemSecondaryAction>
                          <Tooltip title="Feedback was sent" placement="top">
                            {synonym.feedbackScore === 1 ? (<DoneIcon />) : (<CloseIcon />)}
                          </Tooltip>
                        </ListItemSecondaryAction>
                      ) :
                        (
                          <ListItemSecondaryAction>
                            <Tooltip title="Good Recommendation" placement="top">
                              <IconButton onClick={async () => await handleFeedback(synonym, 1)}>
                                <ThumbUpIcon />
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="Bad Recommendation" placement="top">
                              <IconButton edge="end" onClick={async () => await handleFeedback(synonym, 0)}>
                                <ThumbDownIcon />
                              </IconButton>
                            </Tooltip>
                          </ListItemSecondaryAction>
                        )
                      }

                    </ListItem>
                  ))}

                </List>
                <Stack
                  direction="row"
                  justifyContent="center"
                  alignItems="center"
                  spacing={1}>
                  <TextField
                    id="standard-basic"
                    label="Add Synonym" variant="standard"
                    onChange={(event) => setSynonymSuggestion(event.target.value)}
                    value={synonymSuggestion} />
                  <Button
                    variant="outlined"
                    onClick={suggestSynonym}>Suggest</Button>
                </Stack>
              </div>
            )}
          </Box>
        </Box>
      </Container>
    </div>
  );
}

export default App;
