#include "parbcg_getline.h"

/*
    getdelim_calc_new_alloc()

    Helper function for getdelim() to figure out an appropriate new
    allocation size that's not too small or too big.

    These numbers seem to work pretty well for most text files.

    returns the input value if it decides that new allocation block
    would be just too big (the caller should handle this as
    an error).
*/
static
size_t nx_getdelim_get_realloc_size( size_t current_size)
{
    enum {
        k_min_realloc_inc = 32,
        k_max_realloc_inc = 1024,
    };

    if (SSIZE_MAX < current_size) return current_size;

    if (current_size <= k_min_realloc_inc) return current_size + k_min_realloc_inc;

    if (current_size >= k_max_realloc_inc) return current_size + k_max_realloc_inc;

    return current_size * 2;
}



/*
    getdelim_append()

    a helper function for getdelim() that adds a new character to
    the outbuffer, reallocating as necessary to ensure the character
    and a following null terminator can fit

*/
static
int nx_getdelim_append( char** lineptr, size_t* bufsize, size_t count, char ch)
{
    char* tmp = NULL;
    size_t tmp_size = 0;

    // assert the contracts for this functions inputs
    assert( lineptr != NULL);
    assert( bufsize != NULL);

    if (count >= (((size_t) SSIZE_MAX) + 1)) {
        // writing more than SSIZE_MAX to the buffer isn't supported
        return -1;
    }

    tmp = *lineptr;
    tmp_size = tmp ? *bufsize : 0;

    // need room for the character plus the null terminator
    if ((count + 2) > tmp_size) {
        tmp_size = nx_getdelim_get_realloc_size( tmp_size);

        tmp = (char*) realloc( tmp, tmp_size);

        if (!tmp) {
            return -1;
        }
    }

    *lineptr = tmp;
    *bufsize = tmp_size;

    // remember, the reallocation size calculation might not have
    // changed the block size, so we have to check again
    if (tmp && ((count+2) <= tmp_size)) {
        tmp[count++] = ch;
        tmp[count] = 0;
        return 1;
    }

    return -1;
}


/*
    nx_getdelim()

    A getdelim() function modeled on the Linux/POSIX/GNU
    function of the same name.

    Read data into a dynamically resizable buffer until
    EOF or until a delimiter character is found.  The returned
    data will be null terminated (unless there's an error allocating
    memory that prevents it).



    params:

        lineptr -   a pointer to a char* allocated by malloc()
                    (actually any pointer that can legitimately be
                    passed to free()).  *lineptr will be updated
                    by getdelim() if the memory block needs to be
                    reallocated to accommodate the input data.

                    *lineptr can be NULL (though lineptr itself cannot),
                    in which case the function will allocate any necessary
                    buffer.

        n -         a pointer to a size_t object that contains the size of
                    the buffer pointed to by *lineptr (if non-NULL).

                    The size of whatever buff the resulting data is
                    returned in will be passed back in *n

        delim -     the delimiter character.  The function will stop
                    reading one this character is read form the stream.

                    It will be included in the returned data, and a
                    null terminator character will follow it.

        stream -    A FILE* stream object to read data from.

    Returns:

        The number of characters placed in the returned buffer, including
        the delimiter character, but not including the terminating null.

        If no characters are read and EOF is set (or attempting to read
        from the stream on the first attempt caused the eof indication
        to be set), a null terminator will be written to the buffer and
        0 will be returned.

        If an error occurs while reading the stream, a 0 will be returned.
        A null terminator will not necessarily be at the end of the data
        written.

        On the following error conditions, the negative value of the error
        code will be returned:

            ENOMEM:     out of memory
            EOVERFLOW:  SSIZE_MAX character written to te buffer before
                        reaching the delimiter
                        (on Windows, EOVERFLOW is mapped to ERANGE)

         The buffer will not necessarily be null terminated in these cases.


    Notes:

        The returned data might include embedded nulls (if they exist
        in the data stream) - in that case, the return value of the
        function is the only way to reliably determine how much data
        was placed in the buffer.

        If the function returns 0 use feof() and/or ferror() to determine
        which case caused the return.

        If EOF is returned after having written one or more characters
        to the buffer, a normal count will be returned (but there will
        be no delimiter character in the buffer).

        If 0 is returned and ferror() returns a non-zero value,
        the data buffer may not be null terminated.

        In other cases where a negative value is returned, the data
        buffer is not necessarily null terminated and there
        is no reliable means to determining what data in the buffer is
        valid.

        The pointer returned in *lineptr and the buffer size
        returned in *n will be valid on error returns unless
        NULL pointers are passed in for one or more of these
        parameters (in which case the return value will be -EINVAL).

*/
ssize_t nx_getdelim(char **lineptr, size_t *n, int delim, FILE *stream)
{
    ssize_t result = 0;
    char* line = NULL;
    size_t size = 0;
    size_t count = 0;
    int err = 0;
    int ch = 0;

    if (!lineptr || !n) {
        return -EINVAL;
    }

    line = *lineptr;
    size = *n;

    for (;;) {
        ch = fgetc( stream);

        if (ch == EOF) {
            break;
        }

        result = nx_getdelim_append( &line, &size, count, ch);

        // check for error adding to the buffer (ie., out of memory)
        if (result < 0) {
            err = -ENOMEM;
            break;
        }

        ++count;

        // check if we're done because we've found the delimiter
        if ((unsigned char)ch == (unsigned char)delim) {
            break;
        }

        // check if we're passing the maximum supported buffer size
        if (count > SSIZE_MAX) {
            err = -EOVERFLOW;
            break;
        }
    }

    // update the caller's data
    *lineptr = line;
    *n = size;

    // check for various error returns
    if (err != 0) {
        return err;
    }

    if (ferror(stream)) {
        return 0;
    }

    if (feof(stream) && (count == 0)) {
        if (nx_getdelim_append( &line, &size, count, 0) < 0) {
            return -ENOMEM;
        }
    }

    return count;
}




ssize_t nx_getline(char **lineptr, size_t *n, FILE *stream)
{
    return nx_getdelim( lineptr, n, '\n', stream);
}



/*
    versions of getline() and getdelim() that attempt to follow
    POSIX semantics (ie. they set errno on error returns and
    return -1 when the stream error indicator or end-of-file
    indicator is set (ie., ferror() or feof() would return
    non-zero).
*/
ssize_t parbcg_getdelim(char **lineptr, size_t *n, char delim, FILE *stream)
{
    ssize_t retval = nx_getdelim( lineptr, n, delim, stream);

    if (retval < 0) {
        errno = -retval;
        retval = -1;
    }

    if (retval == 0) {
        retval = -1;
    }

    return retval;
}

ssize_t parbcg_getline(char **lineptr, size_t *n, FILE *stream)
{
    return parbcg_getdelim( lineptr, n, '\n', stream);
}
