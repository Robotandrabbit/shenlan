import asyncio
import os
import tempfile
import uuid
from contextlib import ContextDecorator
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, Optional, Type, cast
from unittest.mock import MagicMock, patch

# aiofiles type stubs only suppport python 3.8 or earlier
import aiofiles  # type: ignore
import aiohttp
import aiohttp.typedefs
from aiobotocore.awsrequest import AioAWSResponse
from aiobotocore.endpoint import convert_to_response_dict
from aiohttp.client_reqrep import ClientResponse
from aiohttp.typedefs import RawHeaders
from botocore.awsrequest import AWSResponse
from botocore.model import OperationModel
from moto import mock_s3

from nuplan.common.utils.s3_utils import (
    download_file_from_s3,
    get_async_s3_session,
    upload_file_to_s3,
    upload_file_to_s3_async,
)


# Mock responses needed to make aiobotocore work with moto.
# For full details, see https://github.com/aio-libs/aiobotocore/issues/755
# and https://gist.github.com/giles-betteromics/12e68b88e261402fbe31c2e918ea4168.
class MockAWSResponse(AioAWSResponse):
    """
    Mock AWS response to make aioboto work with moto.
    """

    def __init__(self, response: AWSResponse):
        """
        Wraps moto's mocked AWS response for use with aioboto.
        :param response: Mocked AWS response.
        """
        self._moto_response = response
        self.status_code = response.status_code
        self.raw = MockHttpClientResponse(response)

    async def _content_prop(self) -> bytes:
        """
        Return moto's response from handle used by aioboto.
        :return: Mocked response content.
        """
        response: bytes = self._moto_response.content
        return response

    async def _text_prop(self) -> str:
        """
        Return moto's response from handle used by aioboto.
        :return: Mocked response text.
        """
        response: str = self._moto_response.text
        return response


class MockHttpClientResponse(ClientResponse):
    """
    Mock Http Client response to make aioboto work with moto.
    """

    def __init__(self, response: AWSResponse):
        """
        Wraps moto's mocked client response for use with aioboto.
        :param response: Mocked AWS response.
        """
        read_index = 0

        async def read(n: int = -1) -> bytes:
            """
            Read handler for response contents.
            :param n: Number of bytes to read.
            :return: Bytes read from response content.
            """
            nonlocal read_index
            nonlocal response

            read_response: bytes = response.content[read_index : read_index + n]
            read_index += n

            return read_response

        self.content = MagicMock(aiohttp.StreamReader)
        self.content.read = read
        self.response = response

    @property
    def raw_headers(self) -> RawHeaders:
        """
        Return the headers encoded the way that aioboto expects them.
        :return: Raw response headers.
        """
        return {k.encode("utf-8"): str(v).encode("utf-8") for k, v in self.response.headers.items()}.items()


def convert_to_response_dict_patch(http_response: AWSResponse, operation_model: OperationModel) -> Dict[str, Any]:
    """
    Patch to aioboto method wrapping the Http response.
    :param http_response: AWS response to be wrapped for aioboto.
    :param operation_model: Unmodified parameter.
    :return: A response dictionary generated by aioboto.
    """
    return convert_to_response_dict(MockAWSResponse(http_response), operation_model)  # type: ignore


async def _looks_like_special_case_error_patch(http_response: AWSResponse) -> bool:
    """
    Patch to prevent an error from being thrown when aioboto inspects the Http response.
    :param http_response: Unused.
    :return: Error indicator (always false).
    """
    return False


class mock_async_s3(ContextDecorator):
    """
    Class for mocking S3 that can be used as both a context manager, and function decorator.
    Only works in/on a synchronous function.
    """

    def __init__(self) -> None:
        """
        Context manager setup.
        """
        super().__init__()
        self.patch_special_case_error = patch(
            "aiobotocore.handlers._looks_like_special_case_error", _looks_like_special_case_error_patch
        )
        self.patch_convert_to_response = patch(
            "aiobotocore.endpoint.convert_to_response_dict", convert_to_response_dict_patch
        )
        self.s3_mocker = mock_s3()

    def __enter__(self) -> Type["mock_async_s3"]:
        """
        Context manager enter.
        """
        self.patch_special_case_error.start()
        self.patch_convert_to_response.start()
        self.s3_mocker.start()
        return cast(Type["mock_async_s3"], self)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_traceback: Optional[TracebackType],
    ) -> None:
        """
        Context manager exit.
        Note that we don't have to handle the incoming exception, by returning None, it will be re-raised.
        :param exc_type: Type of any exception that occured.
        :param exc_value: Exception that occured, or None.
        :param traceback: Traceback if an exception occured.
        :return: Always None, so exceptions are never swallowed.
        """
        self.patch_special_case_error.stop()
        self.patch_convert_to_response.stop()
        self.s3_mocker.stop()
        return None


async def create_mock_bucket(bucket_name: str) -> None:
    """
    Create a bucket using async S3 client.
    :param bucket_name: Name to create bucket with.
    """
    session = get_async_s3_session()
    async with session.client("s3") as async_s3_client:
        await async_s3_client.create_bucket(Bucket=bucket_name)


async def setup_mock_s3_directory(
    expected_relative_path_and_contents: Dict[str, str], directory: Path, bucket: str
) -> None:
    """
    Setup download directory with the expected files and structure.
    Assumes that upload_file_to_s3_async works correctly.
    :param expected_relative_path_and_contents: Dictionary mapping paths to contents of expected files.
    :param directory: Directory path to upload to in s3.
    :param bucket: Bucket to upload to in s3.
    """
    await create_mock_bucket(bucket)

    with tempfile.TemporaryDirectory() as temp_dir:
        for relative_path, contents in expected_relative_path_and_contents.items():
            local_path = Path(os.path.join(temp_dir, relative_path))
            local_path.parents[0].mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(local_path, "w") as f:
                await f.write(contents)

            await upload_file_to_s3_async(local_path, directory / relative_path, bucket)


def set_mock_object_from_aws(s3_key: Path, s3_bucket: str) -> None:
    """
    Retrieve an object from real S3 and upload it to mock S3.
    :param s3_key: The S3 key to retrieve and store.
    :param s3_bucket: The S3 bucket to retrieve from and store to. Created if it doesn't exist.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        dump_file = Path(tmp_dir) / f"{str(uuid.uuid4())}.dat"
        download_file_from_s3(dump_file, s3_key, s3_bucket)

        with mock_async_s3():
            _ = get_async_s3_session(force_new=True)

            asyncio.run(create_mock_bucket(s3_bucket))
            upload_file_to_s3(dump_file, s3_key, s3_bucket)
