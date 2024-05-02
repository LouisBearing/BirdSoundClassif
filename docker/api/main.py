from fastapi import FastAPI

app = FastAPI()


@app.get('/test')
def simulate_upload():
    audio_file_path = "turdus_merlula.wav"
    pass




# Upload function from NBM:
"""
@router.post("/upload", response_model=schemas.MediaUploadResponse, status_code=200)
async def upload_audio(
    *,
    db: Session = Depends(deps.get_db),
    audio_file: UploadFile = File(None),
    audio_url: str = Form(None),
    audio_duration: str = Form(None),
    annotations: UploadFile = File(...),
    begin_date: str = Form(None),
    device_id: str = Form(None),
    site_id: str = Form(None),
    file_source: str = Form(...),
    current_user: models.User = Depends(deps.get_current_active_user)
):
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Uploading audio for user: {current_user.id}")
    logger.info(f"\n\nnextcloud user: {settings.NEXTCLOUD_USER}")
    logger.info(f"\n\nnextcloud password: {settings.NEXTCLOUD_PASSWORD}")
    logger.info(f"\n\nnextcloud host: {settings.NEXTCLOUD_HOST}")
    file_url = audio_url
    duration = audio_duration
    meta = json.dumps({})

    # If MEDIA_UPLOAD, upload the file to Nextcloud and set url, duration and metadata afterwards
    if audio_file:
        logger.info(f"Audio file detected for user: {current_user.id}")
        if not settings.NEXTCLOUD_HOST or not settings.NEXTCLOUD_USER or not settings.NEXTCLOUD_PASSWORD:
            logger.error("Nextcloud configuration missing")
            raise HTTPException(status_code=400, detail=[{"type": "next_cloud_config"}])

        # create current user directory in nextcloud
        file_directory = f"mediae/audio/{current_user.id}/"
        try:
            response = requests.request(
                'MKCOL',
                f"{settings.NEXTCLOUD_HOST}/remote.php/dav/files/{settings.NEXTCLOUD_USER}/{file_directory}",
                auth=(settings.NEXTCLOUD_USER, settings.NEXTCLOUD_PASSWORD)
            )
            response.raise_for_status()
            logger.info(f"Created directory in Nextcloud for user: {current_user.id}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to create directory in Nextcloud for user: {current_user.id}. Error: {str(e)}")
            raise HTTPException(status_code=500, detail=[{"type": "nextcloud_directory_creation_fail", "error": str(e)}])

        # make a copy before read info => It's a trick to avoid soundfile read issue on already opened tmp file
        file_unique_id = str(uuid.uuid4())
        new_temp_path = os.path.join(tempfile.gettempdir(), file_unique_id)
        audio_content = await audio_file.read()
        with open(new_temp_path, "wb+") as file_object:
            file_object.write(audio_content)

        # get audio info from file
        try:
            audio_info = soundfile.info(new_temp_path)
            logger.info(f"Extracted audio info for user: {current_user.id}")
        except RuntimeError as e:
            logger.error(f"Invalid audio file for user: {current_user.id}. Error: {str(e)}")
            raise HTTPException(status_code=400, detail=[{"type": "invalid_audio"}])

        os.remove(new_temp_path)

        # upload audio file into nextcloud created directory
        extension = audio_info.format.lower()
        file_name = f"{file_unique_id}.{extension}"

        try:
            response = requests.put(
                f"{settings.NEXTCLOUD_HOST}/remote.php/dav/files/{settings.NEXTCLOUD_USER}/{file_directory}{file_name}",
                data=audio_content,
                auth=(settings.NEXTCLOUD_USER, settings.NEXTCLOUD_PASSWORD)
            )
            response.raise_for_status()
            logger.info(f"Uploaded audio file to Nextcloud for user: {current_user.id}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to upload audio file to Nextcloud for user: {current_user.id}. Error: {str(e)}")
            raise HTTPException(status_code=500, detail=[{"type": "nextcloud_file_upload_fail", "error": str(e)}])

        file_url = f"{file_directory}{file_name}"

        # get metadata to fill media model
        meta = {
            "samplerate": audio_info.samplerate,
            "channels": audio_info.channels,
            "sections": audio_info.sections,
            "format": audio_info.format,
            "subtype": audio_info.subtype
        }

        # get duration to fill media model
        seconds = floor(audio_info.duration)
        microseconds = int((audio_info.duration - seconds) * 100000)
        minutes, seconds = divmod(seconds, 60)
        hour, minutes = divmod(minutes, 60)
        duration = time(hour=hour, minute=minutes, second=seconds, microsecond=microseconds)

    # create media in database
    media_in = schemas.MediaCreate(
        file_url=file_url,
        file_source=file_source,
        begin_date=begin_date,
        device_id=device_id,
        site_id=site_id,
        meta=json.dumps(meta),
        duration=duration
    )

    media = crud.media.create(db=db, obj_in=media_in, created_by=current_user.id)
    logger.info(f"Created media in database for user: {current_user.id}")

    # get information from annotations file
    annotations_content = await annotations.read()
    existing_labels = crud.standardlabel.get_multi(db, limit=9999)
    (medialabel_schemas, invalid_lines) = get_medialabel_schemas_from_file_content(
        annotations_content,
        media.id,
        existing_labels
    )

    # create medialabels in database from previous annotations file processing
    medialabels = []
    for medialabel_schema in medialabel_schemas:
        medialabel = crud.medialabel.create(db=db, obj_in=medialabel_schema, created_by=current_user.id)
        medialabels.append(medialabel)

    logger.info(f"Created medialabels in database for user: {current_user.id}")

    response = schemas.MediaUploadResponse(
        medialabels=medialabels,
        media=media,
        invalid_lines=invalid_lines
    )
    return response



"""