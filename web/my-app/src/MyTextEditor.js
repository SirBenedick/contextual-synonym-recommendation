import React, { useState } from 'react';
import { Button, TextareaAutosize } from '@mui/material';
import { Box } from '@mui/material';
import Alert from '@mui/material/Alert';

const MyTextEditor = (props) => {

    const [message, setMessage] = useState('');
    const [showInformation, setShowInformation] = useState(false);

    const myRefTextArea = React.createRef()

    const clicked = async () => {
        // Get selected word from textarea
        let cursorStart = myRefTextArea.current.selectionStart;
        let cursorEnd = myRefTextArea.current.selectionEnd;
        const selectedWord = message.substring(cursorStart, cursorEnd)

        // Check if selectedWord is a single word
        if (
            selectedWord === "" ||
            selectedWord.includes(" ") ||
            selectedWord.includes(",") ||
            selectedWord.includes(".") ||
            selectedWord.includes("!") ||
            selectedWord.includes("?") ||
            selectedWord.includes("-") ||
            selectedWord.includes("_") ||
            selectedWord.includes("(") ||
            selectedWord.includes(")") ||
            selectedWord.includes("{") ||
            selectedWord.includes("}") ||
            selectedWord.includes("[") ||
            selectedWord.includes("]") ||
            selectedWord.includes("\n")
        ) {
            // TODO: show user an error message
            console.log("Not a single word")
            setShowInformation(true)
            return false
        }
        setShowInformation(false)
        // Replace word with [MASK]
        const firstPartOfSentence = message.substring(0, cursorStart)
        const lastPartOfSentence = message.substring(cursorEnd, message.length)
        const sentenceWithTokenMask = firstPartOfSentence + "[MASK]" + lastPartOfSentence

        props.setSentenceWithTokenMask(sentenceWithTokenMask)
        props.setSelectedWord(selectedWord)
        props.getSynonymsFromAPI(sentenceWithTokenMask, selectedWord)
    }

    const handleTextChange = event => {
        setMessage(event.target.value);
    }


    return (
        <Box
            display="flex"
            flexDirection="column"
            justifyContent="center"
            alignItems="center"
        >
            <TextareaAutosize
                display="flex"
                ref={myRefTextArea}
                autoFocus={true}
                onChange={handleTextChange}
                className='html-editor Sentence-input'
                aria-label="empty textarea"
                placeholder="Type your sentence here..."
            />
            {showInformation && <Alert severity="info">
                Please only select one exact word
            </Alert>}
            <Button
                display
                variant="contained"
                onClick={clicked}
            >
                Get Synonyms Recommendations
            </Button>
        </Box>
    )
}

export default MyTextEditor